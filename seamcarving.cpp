#include <iostream>
#include <luisa-compute.h>
#include <lodepng.h>
#include <vector>

using namespace luisa;
using namespace luisa::compute;

int main(int, char **argv)
{
    Context context(argv[0]);
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream(StreamTag::COMPUTE);

    Callable luminance = [](Float4 rgba) noexcept
    {
        return 0.2126f * rgba.x + 0.7152f * rgba.y + 0.0722f * rgba.z;
    };

    Kernel2D energy_kernel = [&luminance](ImageFloat out, BindlessVar in_bindless, Float2 resolution) noexcept
    {
        Var coord = dispatch_id().xy();
        auto in = in_bindless.tex2d(0u);
        auto dx = luminance((in.sample(make_float2(coord.x + 1.f, coord.y - 1.f) / resolution) + 2.0f * in.sample(make_float2(coord.x + 1.f, coord.y * 1.f) / resolution) + in.sample(make_float2(coord.x + 1.f, coord.y + 1.f) / resolution)) -
                            (in.sample(make_float2(coord.x - 1.f, coord.y - 1.f) / resolution) + 2.0f * in.sample(make_float2(coord.x - 1.f, coord.y * 1.f) / resolution) + in.sample(make_float2(coord.x - 1.f, coord.y + 1.f) / resolution)));
        auto dy = luminance((in.sample(make_float2(coord.x - 1.f, coord.y + 1.f) / resolution) + 2.0f * in.sample(make_float2(coord.x * 1.f, coord.y + 1.f) / resolution) + in.sample(make_float2(coord.x + 1.f, coord.y + 1.f) / resolution)) -
                            (in.sample(make_float2(coord.x - 1.f, coord.y - 1.f) / resolution) + 2.0f * in.sample(make_float2(coord.x * 1.f, coord.y - 1.f) / resolution) + in.sample(make_float2(coord.x + 1.f, coord.y - 1.f) / resolution)));
        out.write(make_uint2(coord.x, coord.y), make_float4(make_float3(luisa::compute::sqrt((dx * dx + dy * dy))), 1.0f));
    };

    Kernel1D cost_kernel = [](ImageFloat cost_map, ImageFloat energy_map, UInt y) noexcept
    {
        Var x = dispatch_id().x;
        Float lt = std::numeric_limits<float>::infinity();
        Float t = cost_map.read(make_uint2(x, y - 1)).x;
        Float rt = std::numeric_limits<float>::infinity();
        $if(x > 0)
        {
            lt = cost_map.read(make_uint2(x - 1, y - 1)).x;
        };
        $if(x + 1 < dispatch_size_x())
        {
            rt = cost_map.read(make_uint2(x + 1, y - 1)).x;
        };
        Var cost = energy_map.read(make_uint2(x, y)).x + min(min(lt, t), rt);
        cost_map.write(make_uint2(x, y), make_float4(make_float3(cost), 1.0f));
    };
    
    auto energy_shader = device.compile(energy_kernel);
    auto cost_shader = device.compile(cost_kernel);

    std::vector<unsigned char> image_png_buffer = {};
    unsigned int error = lodepng::load_file(image_png_buffer, "/home/tianyu/seamcarving.png");
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

    auto delete_vertial_seam = [&](std::vector<unsigned char> image_buffer, uint width, uint height) -> std::vector<unsigned char>
    {
        // Begin device commands.
        Image<float> image = device.create_image<float>(PixelStorage::BYTE4, width, height, 0u);
        stream << image.copy_from(image_buffer.data()) << synchronize();
        Image<float> image_energy = device.create_image<float>(PixelStorage::FLOAT4, width, height, 0u);
        BindlessArray bindless = device.create_bindless_array(1u);
        bindless.emplace_on_update(0u, image, Sampler(Sampler::Filter::LINEAR_LINEAR, Sampler::Address::MIRROR));
        Image<float> image_cost = device.create_image<float>(PixelStorage::FLOAT4, width, height, 0u);
        stream << bindless.update() << energy_shader(image_energy, bindless, make_float2(width, height)).dispatch(width, height) << image_cost.copy_from(image_energy) << synchronize();
        for (uint y = 1; y < height; ++y)
            stream << cost_shader(image_cost, image_energy, y).dispatch(width) << synchronize();
        std::vector<float4> download_image_cost(width * height);
        stream << image_cost.copy_to(download_image_cost.data()) << synchronize();
        // End device commands.

        // Begin find seam.
        uint trace_start = 0u;
        float min_value = std::numeric_limits<float>::infinity();
        for (uint i = 0; i < width; ++i)
        {
            if (download_image_cost[(height - 1) * width + i].x < min_value)
            {
                min_value = download_image_cost[(height - 1) * width + i].x;
                trace_start = i;
            }
        }
        std::cout << "Trace start:" << trace_start << std::endl;
        std::vector<uint> seam = {};
        seam.push_back(trace_start);
        for (uint i = height - 1; i > 0; --i)
        {
            const auto &current_x = seam.back();
            float lt = std::numeric_limits<float>::infinity();
            float rt = std::numeric_limits<float>::infinity();
            float t = download_image_cost[(i - 1) * width + current_x].x;
            if (current_x > 0)
            {
                lt = download_image_cost[(i - 1) * width + current_x - 1].x;
            }
            if (current_x + 1 < width)
            {
                rt = download_image_cost[(i - 1) * width + current_x + 1].x;
            }
            uint current_x_candidate = current_x;
            if (lt < t)
            {
                t = lt;
                current_x_candidate = current_x - 1;
            }
            if (rt < t)
            {
                current_x_candidate = current_x + 1;
            }
            seam.push_back(current_x_candidate);
        }
        // End find seam.

        // Begin new image creation.
        std::vector<unsigned char> result_image;
        uint new_width = width - 1;
        result_image.resize(new_width * height * 4u);
#pragma omp parallel for
        for (uint i = 0; i < height; ++i)
        {
            uint jj = 0;
            for (uint j = 0; j < width; ++j)
            {
                if (seam[height - 1 - i] != j)
                {
                    result_image[(i * new_width + jj) * 4u + 0u] = image_buffer[(i * width + j) * 4u + 0u];
                    result_image[(i * new_width + jj) * 4u + 1u] = image_buffer[(i * width + j) * 4u + 1u];
                    result_image[(i * new_width + jj) * 4u + 2u] = image_buffer[(i * width + j) * 4u + 2u];
                    result_image[(i * new_width + jj) * 4u + 3u] = image_buffer[(i * width + j) * 4u + 3u];
                    ++jj;
                }
            }
        }
        // End new image creation.

        return result_image;
    };

    auto result_image = delete_vertial_seam(image_buffer, width, height);

    std::vector<unsigned char>
        image_save_buffer = {};
    lodepng::State state;
    lodepng::encode(image_save_buffer, result_image, width - 1, height, state);
    lodepng::save_file(image_save_buffer, "./sc.png");
}