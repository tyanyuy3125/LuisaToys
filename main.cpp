#include <iostream>

#include "luisa-compute.h"

using namespace luisa;
using namespace luisa::compute;

#include <unistd.h>
[[nodiscard]] auto get_current_exe_path() noexcept {
    char pathbuf[PATH_MAX] = {};
    for (auto p : {"/proc/self/exe", "/proc/curproc/file", "/proc/self/path/a.out"}) {
        if (auto size = readlink(p, pathbuf, sizeof(pathbuf)); size > 0) {
            luisa::string_view path{pathbuf, static_cast<size_t>(size)};
            return std::filesystem::canonical(path).string();
        }
    }
}

int main(int, char** argv){
    Context context{argv[0]};
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
}
