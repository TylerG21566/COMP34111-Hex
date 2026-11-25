#include "evaluation/Network.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace hex {

namespace {

// The file format, shared with export.py
constexpr char kMagic[] = {'H', 'E', 'X', 'N'};
constexpr std::uint32_t kVersion = 1;
enum LayerType : std::uint32_t { kLinear = 1, kRelu = 2, kTanh = 3 };

static_assert(std::endian::native == std::endian::little, "the file is little-endian");

// Reads values from the bytes of a file, and throws if there are too few
class Reader {
public:
    explicit Reader(std::vector<char> bytes) : bytes_(std::move(bytes)) {}

    std::uint32_t readInt() {
        std::uint32_t value;
        copy(&value, sizeof value);
        return value;
    }

    std::vector<float> readFloats(std::uint64_t count) {
        if (count > remaining() / sizeof(float)) throw std::runtime_error("it is too short");
        std::vector<float> values(count);
        copy(values.data(), count * sizeof(float));
        for (float v : values) {
            if (!std::isfinite(v)) throw std::runtime_error("it has weights that are not numbers");
        }
        return values;
    }

    bool startsWith(const char* text, std::size_t size) {
        if (remaining() < size) return false;
        const bool same = std::memcmp(bytes_.data() + position_, text, size) == 0;
        position_ += size;
        return same;
    }

    std::size_t remaining() const { return bytes_.size() - position_; }

private:
    void copy(void* to, std::size_t size) {
        if (size > remaining()) throw std::runtime_error("it is too short");
        std::memcpy(to, bytes_.data() + position_, size);
        position_ += size;
    }

    std::vector<char> bytes_;
    std::size_t position_ = 0;
};

Linear readLinear(Reader& reader) {
    constexpr std::uint32_t kMaxSize = 1 << 20;
    const std::uint32_t inputs = reader.readInt(), outputs = reader.readInt();
    if (inputs == 0 || outputs == 0 || inputs > kMaxSize || outputs > kMaxSize) {
        throw std::runtime_error("it has a layer with an invalid size");
    }

    // PyTorch stores weights as [output][input]
    const std::vector<float> stored = reader.readFloats(std::uint64_t{inputs} * outputs);
    Linear linear;
    linear.inputs = static_cast<int>(inputs);
    linear.outputs = static_cast<int>(outputs);
    linear.weights.resize(stored.size());
    for (std::size_t o = 0; o < outputs; ++o) {
        for (std::size_t i = 0; i < inputs; ++i) {
            linear.weights[i * outputs + o] = stored[o * inputs + i];
        }
    }
    linear.bias = reader.readFloats(outputs);
    return linear;
}

Layers readLayers(Reader& reader) {
    Layers layers;
    const std::uint32_t count = reader.readInt();
    for (std::uint32_t i = 0; i < count; ++i) {
        switch (reader.readInt()) {
            case kLinear: layers.push_back(readLinear(reader)); break;
            case kRelu: layers.push_back(Relu{}); break;
            case kTanh: layers.push_back(Tanh{}); break;
            default: throw std::runtime_error("it has a kind of layer that is not supported");
        }
    }
    return layers;
}

// Checks that each layer fits the one before. Returns the number of outputs.
int checkLayers(const Layers& layers, int size, const std::string& name) {
    for (const Layer& layer : layers) {
        const Linear* linear = std::get_if<Linear>(&layer);
        if (linear == nullptr) continue;
        if (linear->inputs != size) {
            throw std::invalid_argument("the " + name + " has a layer with " +
                                        std::to_string(linear->inputs) + " inputs, expected " +
                                        std::to_string(size));
        }
        const std::size_t weights = static_cast<std::size_t>(linear->inputs) * linear->outputs;
        if (linear->weights.size() != weights ||
            linear->bias.size() != static_cast<std::size_t>(linear->outputs)) {
            throw std::invalid_argument("the " + name + " has the wrong number of weights");
        }
        size = linear->outputs;
    }
    return size;
}

std::vector<float> run(const Linear& linear, const std::vector<float>& input) {
    std::vector<float> output = linear.bias;
    const int outputs = linear.outputs;
    float* out = output.data();
    for (int i = 0; i < linear.inputs; ++i) {
        // Most board cells, and many values after a ReLU, are 0
        const float value = input[i];
        if (value == 0.0f) continue;
        const float* weights = linear.weights.data() + static_cast<std::size_t>(i) * outputs;
        for (int o = 0; o < outputs; ++o) out[o] += value * weights[o];
    }
    return output;
}

std::vector<float> run(const Relu&, std::vector<float> values) {
    for (float& v : values) v = std::max(v, 0.0f);
    return values;
}

std::vector<float> run(const Tanh&, std::vector<float> values) {
    for (float& v : values) v = std::tanh(v);
    return values;
}

std::vector<float> runLayers(const Layers& layers, std::vector<float> values) {
    for (const Layer& layer : layers) {
        values = std::visit([&](const auto& l) { return run(l, std::move(values)); }, layer);
    }
    return values;
}

}  // namespace

Network::Network(Layers trunk, Layers valueHead, Layers policyHead)
    : trunk_(std::move(trunk)),
      valueHead_(std::move(valueHead)),
      policyHead_(std::move(policyHead)) {
    const int features = checkLayers(trunk_, kInputs, "trunk");
    if (checkLayers(valueHead_, features, "value head") != 1) {
        throw std::invalid_argument("the value head must have 1 output");
    }
    if (valueHead_.empty() || !std::holds_alternative<Tanh>(valueHead_.back())) {
        throw std::invalid_argument("the value head must end in tanh");
    }
    if (checkLayers(policyHead_, features, "policy head") != kCells) {
        throw std::invalid_argument("the policy head must have " + std::to_string(kCells) +
                                    " outputs");
    }
}

Network Network::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open network file " + path);
    Reader reader(std::vector<char>(std::istreambuf_iterator<char>(file), {}));

    try {
        if (!reader.startsWith(kMagic, sizeof kMagic)) {
            throw std::runtime_error("it was not written by export.py");
        }
        if (reader.readInt() != kVersion) {
            throw std::runtime_error("it has a different format version");
        }
        Layers trunk = readLayers(reader);
        Layers valueHead = readLayers(reader);
        Layers policyHead = readLayers(reader);
        if (reader.remaining() != 0) throw std::runtime_error("it has extra data at the end");
        return Network(std::move(trunk), std::move(valueHead), std::move(policyHead));
    } catch (const std::exception& e) {
        throw std::runtime_error("invalid network file " + path + ": " + e.what());
    }
}

Network::Output Network::forward(const std::vector<float>& input) const {
    const std::vector<float> features = runLayers(trunk_, input);

    Output output;
    output.value = runLayers(valueHead_, features)[0];
    output.policy = runLayers(policyHead_, features);

    // Softmax. Subtracting the largest logit keeps exp() from overflowing.
    const float largest = *std::max_element(output.policy.begin(), output.policy.end());
    float total = 0.0f;
    for (float& p : output.policy) {
        p = std::exp(p - largest);
        total += p;
    }
    for (float& p : output.policy) p /= total;
    return output;
}

}  // namespace hex
