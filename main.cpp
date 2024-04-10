#include <iostream>

#include <luisa-compute.h>
#include <lodepng.h>

using namespace luisa;
using namespace luisa::compute;

Float4 extCallable()
{
    return make_float4(0.0f, 0.0f, 1.0f, 1.0f);
}

int main(int, char** argv){
    Context context{argv[0]};
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();

    Image<float> device_image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(1024u, 1024u), 0u);

    Callable lambda_callable = []() noexcept -> Float4 {
        // return make_float4(0.0f, 1.0f, 0.0f, 1.0f);
        return extCallable();
    };

    Kernel2D kernel = [&](Var<Image<float>> image) noexcept {
        Var coord = dispatch_id().xy();
        // image->write(coord, make_float4(1.0f, 0.0f, 0.0f, 1.0f));
        image->write(coord, lambda_callable());
    };

    auto fill_image = device.compile(kernel);
    std::vector<unsigned char> downloaded_image(1024u * 1024u * 4u);
    stream << fill_image(device_image.view(0)).dispatch(1024u, 1024u) << device_image.copy_to(downloaded_image.data()) << synchronize();

    lodepng::State state;
    std::vector<unsigned char> png;
    unsigned error = lodepng::encode(png, downloaded_image, 1024u, 1024u, state);
    if(!error) lodepng::save_file(png, "output.png");

    //if there's an error, display it
    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
}
