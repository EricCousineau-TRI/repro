#include <cstdlib>
#include <unistd.h>
#include <stdio.h>

int main() {
//    char* const args[] = {"bash", "-c", "test -d bazel-bin && cd ./bazel-bin/noddy_test.runfiles/clion_python_debug_shared && pwd", nullptr};
//    execvp(args[0], args);
    if (chdir("./bazel-bin/noddy_test.runfiles/clion_python_debug_shared") != 0) {
        printf("Already in dir\n");
    }
    system("which python");
//    system("env");
    return system("python ./noddy_test");
}
