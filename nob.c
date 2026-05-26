#define NOB_IMPLEMENTATION
#include "nob.h"
#define FLAG_IMPLEMENTATION
#include "flag.h"

Cmd cmd = {0};

static void usage(void)
{
    fprintf(stderr, "Usage: %s [<FLAGS>] [--] [<program args>]\n", flag_program_name());
    fprintf(stderr, "FLAGS:\n");
    flag_print_options(stderr);
}

int main(int argc, char **argv)
{
    GO_REBUILD_URSELF(argc, argv);

    bool run = false;
    bool help = false;
    flag_bool_var(&run, "run", false, "Run the program after compilation.");
    flag_bool_var(&help, "help", false, "Print this help message.");

    if (!flag_parse(argc, argv)) {
        usage();
        flag_print_error(stderr);
        return 1;
    }

    if (help) {
        usage();
        return 0;
    }

    cmd_append(&cmd, "cc");
    cmd_append(&cmd, "-Wall");
    cmd_append(&cmd, "-Wextra");
    cmd_append(&cmd, "-ggdb");
#if defined(__APPLE__)
    cmd_append(&cmd, "-I/opt/homebrew/opt/raylib/include/");
#else
    cmd_append(&cmd, "-I./raylib-5.5_linux_amd64/include/");
#endif
    cmd_append(&cmd, "-o", "./main", "main.c");
#if defined(__APPLE__)
    cmd_append(&cmd, "-L/opt/homebrew/opt/raylib/lib/");
    cmd_append(&cmd, "-lraylib");
    cmd_append(&cmd, "-framework", "OpenGL");
    cmd_append(&cmd, "-framework", "Cocoa");
    cmd_append(&cmd, "-framework", "IOKit");
    cmd_append(&cmd, "-framework", "CoreVideo");
#else
    cmd_append(&cmd, "-L./raylib-5.5_linux_amd64/lib/");
    cmd_append(&cmd, "-l:libraylib.a");
    cmd_append(&cmd, "-lm");
#endif
    if (!cmd_run(&cmd)) return 1;

    if (run) {
        cmd_append(&cmd, "./main");
        da_append_many(&cmd, flag_rest_argv(), flag_rest_argc());
        if (!cmd_run(&cmd)) return 1;
    }

    return 0;
}
