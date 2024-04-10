#include <iostream>

#include <luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

int main(int, char** argv){
    Context context{argv[0]};
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    uint2 resolution = make_uint2(800u, 600u);
    Window window("Test Window", resolution);
    Swapchain swpchain = device.create_swapchain(stream, SwapchainOption{
        .display = window.native_display(),
        .window = window.native_handle(),
        .size = resolution,
        .wants_hdr = false,
        .wants_vsync = false,
        .back_buffer_count = 2u
    });
    Image<float> display = device.create_image<float>(swpchain.backend_storage(), resolution);

    Kernel2D kernel = [&](ImageFloat image, Float time) noexcept {
        Var coord = dispatch_id().xy();
        Var screen_coord = make_float2(coord) / make_float2(resolution);
        Var initial_amp = make_float3(screen_coord, 1.0f);
        // (cos(phi) + 1) / 2 = initial_amp
        Var initial_phase = acos(initial_amp * 2.0f - 1.0f);
        Var wt_phi = 2.0f * pi / 1000.0f * time + initial_phase;
        image.write(coord, make_float4((cos(wt_phi) + 1.0f) / 2.0f, 1.0f));
    };
    auto shader = device.compile(kernel);

    Clock clock;
    clock.tic();
    while(!window.should_close())
    {
        window.poll_events();
        CommandList cmds;
        cmds.reserve(1u, 0u);
        stream << shader(display, clock.toc()).dispatch(resolution) << swpchain.present(display);
    }
    stream << synchronize();
}
