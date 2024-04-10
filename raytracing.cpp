#include <luisa-compute.h>
#include <lodepng.h>
#include <tiny_obj_loader.h>

#include <iostream>
#include <map>
#include <vector>
#include <set>

using namespace luisa::compute;

struct Compare
{
    inline bool operator()(const tinyobj::index_t &a,
                           const tinyobj::index_t &b)
    {
        if (a.vertex_index < b.vertex_index)
            return true;
        if (a.vertex_index > b.vertex_index)
            return false;

        if (a.normal_index < b.normal_index)
            return true;
        if (a.normal_index > b.normal_index)
            return false;

        if (a.texcoord_index < b.texcoord_index)
            return true;
        if (a.texcoord_index > b.texcoord_index)
            return false;

        return false;
    }
};

auto loadObj(const std::string &path)
{
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path))
    {
        if (!reader.Error().empty())
        {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty())
    {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();
    auto &materials = reader.GetMaterials();

    std::vector<Mesh> ret_mesh = {};

    for (const auto &shape : shapes)
    {
        std::set<int> materialIDs;
        for(auto faceMatID : shape.mesh.material_ids)
        {
            materialIDs.insert(faceMatID);
        }

        for(int materialID : materialIDs)
        {
            std::map<tinyobj::index_t, int, Compare> knownVertices;
            std::map<std::string, int> knownTextures;
            Mesh mesh = {};

            
        }
    }
}

int main(int, char **argv)
{
    Context context{argv[0]};
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();

    
    AccelOption accel_option;
    accel_option.allow_compaction = false;
    accel_option.allow_update = false;
    accel_option.hint = AccelOption::UsageHint::FAST_TRACE;
    Accel accel = device.create_accel(accel_option);
}