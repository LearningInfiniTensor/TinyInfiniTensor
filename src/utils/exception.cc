#include "utils/exception.h"

namespace infini {
Exception::Exception(const std::string &msg) : std::runtime_error(msg) {}
} // namespace infini
