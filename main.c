    #include <raylib.h>
    #include <raymath.h>
    #include <stdlib.h>
    #include <dirent.h>
    #include <string.h>
    #include <float.h>
    #include "nob.h"

    // --- constants --------------------------------------------------------------

    #define W             900
    #define H             700
    #define PAD           60
    #define POINT_RADIUS  4
    #define DBSCAN_EPS    30.0f
    #define DBSCAN_MINPTS 5
    #define NOISE        -1
    #define UNCLASSIFIED  0
    #define THUMB         64      // resize each image to 64x64 before feature extraction
    #define TRAIN_RATIO   0.8f    // 80% train, 20% test


    static Color CLUSTER_COLORS[] = {
        { 214,  39,  40, 200 },
        {  31, 119, 180, 200 },
        {  44, 160,  44, 200 },
        { 148, 103, 189, 200 },
        { 255, 127,  14, 200 },
        {  23, 190, 207, 200 },
    };
    #define N_COLORS (int)(sizeof(CLUSTER_COLORS)/sizeof(CLUSTER_COLORS[0]))

    // --- types ------------------------------------------------------------------

    typedef enum { CLASS_KIRMIZI = 0, CLASS_SIIRT = 1 } GroundTruth;

    typedef struct {
        Vector2    pos;        // screen position derived from (hue, sat)
        float      hue, sat;  // raw features: avg hue 0-360, avg sat 0-1
        GroundTruth truth;     // actual class
        int        cluster;    // DBSCAN result (only set for train points)
        bool       visited;
        bool       is_test;    // true = held-out validation point
        GroundTruth predicted; // what our classifier guessed (test points only)
    } ImgPoint;

    typedef struct { ImgPoint *items; size_t count; size_t capacity; } ImgPoints;
    typedef struct { size_t   *items; size_t count; size_t capacity; } Index;

    // --- feature extraction -----------------------------------------------------

    Vector2 extract_features(const char *path)
    {
        Image img = LoadImage(path);
        if (!img.data) return (Vector2){-1, -1};
        ImageResize(&img, THUMB, THUMB);
        ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
        Color *px = LoadImageColors(img);
        float sh = 0, ss = 0;
        int   n  = THUMB * THUMB;
        for (int i = 0; i < n; i++) {
            Vector3 hsv = ColorToHSV(px[i]);
            sh += hsv.x;
            ss += hsv.y;
        }
        UnloadImageColors(px);
        UnloadImage(img);
        return (Vector2){sh / n, ss / n};
    }

    static float hue_min, hue_max, sat_min, sat_max;

    void compute_range(ImgPoints *pts)
    {
        hue_min = hue_max = pts->items[0].hue;
        sat_min = sat_max = pts->items[0].sat;
        for (size_t i = 1; i < pts->count; i++) {
            if (pts->items[i].hue < hue_min) hue_min = pts->items[i].hue;
            if (pts->items[i].hue > hue_max) hue_max = pts->items[i].hue;
            if (pts->items[i].sat < sat_min) sat_min = pts->items[i].sat;
            if (pts->items[i].sat > sat_max) sat_max = pts->items[i].sat;
        }
        float hpad = (hue_max - hue_min) * 0.05f;
        float spad = (sat_max - sat_min) * 0.05f;
        hue_min -= hpad; hue_max += hpad;
        sat_min -= spad; sat_max += spad;
    }

    Vector2 to_screen(float hue, float sat)
    {
        float x = PAD + (hue - hue_min) / (hue_max - hue_min) * (W - 2*PAD);
        float y = (H - PAD) - (sat - sat_min) / (sat_max - sat_min) * (H - 2*PAD);
        return (Vector2){x, y};
    }

    void load_folder(const char *folder, GroundTruth truth, ImgPoints *pts)
    {
        DIR *d = opendir(folder);
        if (!d) { TraceLog(LOG_ERROR, "Cannot open: %s", folder); return; }
        struct dirent *e;
        while ((e = readdir(d))) {
            if (e->d_name[0] == '.') continue;
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s", folder, e->d_name);
            Vector2 f = extract_features(path);
            if (f.x < 0) continue;
            ImgPoint p = {
                .hue = f.x, .sat = f.y,
                .truth = truth,
                .cluster = UNCLASSIFIED,
            };
            da_append(pts, p);
        }
        closedir(d);
    }

    // --- train/test split -------------------------------------------------------

    void shuffle(ImgPoints *pts)
    {
        for (size_t i = pts->count - 1; i > 0; i--) {
            size_t j = rand() % (i + 1);
            ImgPoint tmp  = pts->items[i];
            pts->items[i] = pts->items[j];
            pts->items[j] = tmp;
        }
    }

    // --- DBSCAN -----------------------------------------------------------------

    void get_neighbors(ImgPoints *pts, size_t target, float eps, Index *out)
    {
        out->count = 0;
        for (size_t i = 0; i < pts->count; i++) {
            if (i == target || pts->items[i].is_test) continue;
            if (Vector2Distance(pts->items[i].pos, pts->items[target].pos) <= eps)
                da_append(out, i);
        }
    }

    int dbscan(ImgPoints *pts, float eps, size_t min_pts)
    {
        for (size_t i = 0; i < pts->count; i++) {
            if (pts->items[i].is_test) continue;
            pts->items[i].visited = false;
            pts->items[i].cluster = UNCLASSIFIED;
        }

        int   cluster_id = 0;
        Index seeds = {0}, nbors = {0};

        for (size_t i = 0; i < pts->count; i++) {
            if (pts->items[i].is_test || pts->items[i].visited) continue;
            pts->items[i].visited = true;

            get_neighbors(pts, i, eps, &nbors);
            if (nbors.count < min_pts) { pts->items[i].cluster = NOISE; continue; }

            cluster_id++;
            pts->items[i].cluster = cluster_id;

            seeds.count = 0;
            for (size_t j = 0; j < nbors.count; j++) da_append(&seeds, nbors.items[j]);

            for (size_t j = 0; j < seeds.count; j++) {
                size_t idx = seeds.items[j];
                if (!pts->items[idx].visited) {
                    pts->items[idx].visited = true;
                    get_neighbors(pts, idx, eps, &nbors);
                    if (nbors.count >= min_pts)
                        for (size_t k = 0; k < nbors.count; k++) da_append(&seeds, nbors.items[k]);
                }
                if (pts->items[idx].cluster == UNCLASSIFIED || pts->items[idx].cluster == NOISE)
                    pts->items[idx].cluster = cluster_id;
            }
        }
        return cluster_id;
    }

    // --- classification & evaluation --------------------------------------------
    GroundTruth *cluster_majority(ImgPoints *pts, int num_clusters)
    {
        GroundTruth *labels = malloc(num_clusters * sizeof(GroundTruth));
        for (int c = 1; c <= num_clusters; c++) {
            int k = 0, s = 0;
            for (size_t i = 0; i < pts->count; i++) {
                if (pts->items[i].is_test || pts->items[i].cluster != c) continue;
                pts->items[i].truth == CLASS_KIRMIZI ? k++ : s++;
            }
            labels[c - 1] = k >= s ? CLASS_KIRMIZI : CLASS_SIIRT;
            TraceLog(LOG_INFO, "Cluster %d → %s  (Kirmizi=%d  Siirt=%d)",
                    c, k >= s ? "Kirmizi" : "Siirt", k, s);
        }
        return labels;
    }


    GroundTruth predict(ImgPoints *pts, GroundTruth *clabels, Vector2 pos)
    {
        float min_d = FLT_MAX;
        int   best  = -1;
        for (size_t i = 0; i < pts->count; i++) {
            if (pts->items[i].is_test || pts->items[i].cluster <= 0) continue;
            float d = Vector2Distance(pts->items[i].pos, pos);
            if (d < min_d) { min_d = d; best = pts->items[i].cluster; }
        }
        if (best <= 0) return CLASS_KIRMIZI;
        return clabels[best - 1];
    }

    void evaluate(ImgPoints *pts, GroundTruth *clabels, float *acc_out)
    {
        int correct = 0, total = 0;
        for (size_t i = 0; i < pts->count; i++) {
            if (!pts->items[i].is_test) continue;
            pts->items[i].predicted = predict(pts, clabels, pts->items[i].pos);
            if (pts->items[i].predicted == pts->items[i].truth) correct++;
            total++;
        }
        *acc_out = total > 0 ? 100.0f * correct / total : 0.0f;
        TraceLog(LOG_INFO, "Test accuracy: %d / %d = %.1f%%", correct, total, *acc_out);
    }

    // --- main -------------------------------------------------------------------

    int main(int argc, char **argv)
    {
        if (argc != 3) {
            fprintf(stderr, "Usage: %s <kirmizi_path> <siirt_path>\n", argv[0]);
            return 1;
        }
        const char *kirmizi_path = argv[1];
        const char *siirt_path   = argv[2];

        srand(42);

        TraceLog(LOG_INFO, "Loading Kirmizi...");
        ImgPoints pts = {0};
        load_folder(kirmizi_path, CLASS_KIRMIZI, &pts);
        TraceLog(LOG_INFO, "Loading Siirt...");
        load_folder(siirt_path, CLASS_SIIRT, &pts);
        TraceLog(LOG_INFO, "Done — %zu images total.", pts.count);

        compute_range(&pts);
        for (size_t i = 0; i < pts.count; i++)
            pts.items[i].pos = to_screen(pts.items[i].hue, pts.items[i].sat);
        shuffle(&pts);
        size_t test_start = (size_t)(pts.count * TRAIN_RATIO);
        for (size_t i = test_start; i < pts.count; i++) pts.items[i].is_test = true;
        TraceLog(LOG_INFO, "Train: %zu   Test: %zu", test_start, pts.count - test_start);

        float        eps       = DBSCAN_EPS;
        int          n_clust   = 0;
        float        accuracy  = 0.0f;
        GroundTruth *clabels   = NULL;
        bool         show_truth = false;
        bool         show_test  = true;

        InitWindow(W, H, "Pistachio DBSCAN classifier");
        SetTargetFPS(60);

        while (!WindowShouldClose()) {
            if (IsKeyPressed(KEY_SPACE)) {
                if (clabels) { free(clabels); clabels = NULL; }
                n_clust = dbscan(&pts, eps, DBSCAN_MINPTS);
                clabels = cluster_majority(&pts, n_clust);
                evaluate(&pts, clabels, &accuracy);
            }
            if (IsKeyPressed(KEY_EQUAL)) eps += 5.0f;
            if (IsKeyPressed(KEY_MINUS) && eps > 5.0f) eps -= 5.0f;
            if (IsKeyPressed(KEY_T)) show_truth = !show_truth;
            if (IsKeyPressed(KEY_V)) show_test  = !show_test;

            BeginDrawing();
            ClearBackground(RAYWHITE);

            DrawLine(PAD, PAD, PAD, H-PAD, LIGHTGRAY);
            DrawLine(PAD, H-PAD, W-PAD, H-PAD, LIGHTGRAY);

            for (size_t i = 0; i < pts.count; i++) {
                ImgPoint *p = &pts.items[i];
                if (p->is_test && !show_test) continue;

                Color c;
                if (show_truth) {
                    c = p->truth == CLASS_KIRMIZI
                        ? (Color){214, 39, 40, 200}
                        : (Color){ 31, 119, 180, 200};
                } else {
                    if      (p->cluster == UNCLASSIFIED) c = (Color){180,180,180,160};
                    else if (p->cluster == NOISE)        c = (Color){210,210,210,160};
                    else c = CLUSTER_COLORS[(p->cluster - 1) % N_COLORS];
                }

                if (p->is_test) {
                    Color outline = (clabels && p->predicted != p->truth)
                        ? (Color){214, 39, 40, 255} : c;
                    DrawCircleLines((int)p->pos.x, (int)p->pos.y, POINT_RADIUS + 2, outline);
                } else {
                    DrawCircleV(p->pos, POINT_RADIUS, c);
                }
            }

            DrawText(TextFormat("eps %.0f  clusters %d  acc %.1f%%", eps, n_clust, accuracy),
                PAD, H-PAD+20, 13, GRAY);

            EndDrawing();
        }

        if (clabels) free(clabels);
        CloseWindow();
        return 0;
    }
