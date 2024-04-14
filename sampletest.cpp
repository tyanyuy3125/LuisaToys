#include <luisa-compute.h>
#include <iostream>
#include <lodepng.h>

int main(int, char **argv)
{
    luisa::compute::Context context{argv[0]};
    luisa::compute::Device device = context.create_device("cuda");
    luisa::compute::Stream stream = device.create_stream(luisa::compute::StreamTag::GRAPHICS);

    std::vector<unsigned char> image_png_buffer = {};
    unsigned int error = lodepng::load_file(image_png_buffer, "/home/tianyu/testimage.png");
    std::vector<unsigned char> image_buffer = {};
    unsigned int width, height;
    if (!error)
        error = lodepng::decode(image_buffer, width, height, image_png_buffer);
    if (error)
    {
        std::cerr << lodepng_error_text(error) << std::endl;
        exit(1);
    }
    std::cout << "Image loaded. Width: " << width << ", height: " << height << ".\n";

    luisa::compute::Image<float> image = device.create_image<float>(luisa::compute::PixelStorage::BYTE4, width, height, 0u);
    stream << image.copy_from(image_buffer.data()) << luisa::compute::synchronize();

    luisa::compute::Kernel2D pixel_level_process = [&](luisa::compute::ImageFloat image) noexcept
    {
        luisa::compute::Var coord = luisa::compute::dispatch_id().xy();
        luisa::compute::Var pixel_color = image->read(coord);
        luisa::compute::Var grayscale = luisa::compute::dot(pixel_color, luisa::compute::make_float4(0.2126f, 0.7152f, 0.0722f, 0.0f));
        image->write(coord, luisa::compute::make_float4(luisa::compute::make_float3(grayscale), 1.0f));
    };

    auto shader = device.compile(pixel_level_process);
    stream << shader(image).dispatch(width, height) << luisa::compute::synchronize();

    auto window_resolution = luisa::compute::make_uint2(400u, 400u);
    luisa::compute::Window window("Display", window_resolution);
    luisa::compute::Swapchain swapchain = device.create_swapchain(stream, luisa::compute::SwapchainOption{
                                                                              .back_buffer_count = 2u,
                                                                              .display = window.native_display(),
                                                                              .window = window.native_handle(),
                                                                              .wants_hdr = false,
                                                                              .wants_vsync = false,
                                                                              .size = window_resolution});

    luisa::compute::Image<float> display = device.create_image<float>(luisa::compute::PixelStorage::BYTE4, window_resolution, 0u);
    luisa::compute::BindlessArray bindless = device.create_bindless_array(1u);
    bindless.emplace_on_update(0u, image, luisa::compute::Sampler(luisa::compute::Sampler::Filter::LINEAR_LINEAR, luisa::compute::Sampler::Address::REPEAT));
    luisa::compute::Kernel2D image_sampling_kernel = [&](luisa::compute::BindlessVar from, luisa::compute::ImageFloat to) noexcept {
        luisa::compute::Var coord = luisa::compute::dispatch_id().xy();
        luisa::compute::Var normalized_coord = luisa::compute::make_float2(coord) / luisa::compute::make_float2(window_resolution);
        to.write(coord, from.tex2d(0u).sample(normalized_coord));
    };

    stream << bindless.update() << device.compile(image_sampling_kernel)(bindless, display).dispatch(window_resolution) << luisa::compute::synchronize();

    while (!window.should_close())
    {
        window.poll_events();

        stream << swapchain.present(display);
    }
    stream << luisa::compute::synchronize();

    return 0;
}