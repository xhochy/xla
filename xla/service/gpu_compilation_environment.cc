/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"
#include "xla/service/compilation_environments.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"   // IWYU pragma: keep
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace xla {

// Default env is currently empty as DebugOptions is already set by default for
// uninitialized flags. This will be changes when we migrate all GPU relevant
// flags to DebugOptions.
std::unique_ptr<GpuCompilationEnvironment> CreateDefaultGpuCompEnv() {
  return std::make_unique<GpuCompilationEnvironment>();
}

StatusOr<std::unique_ptr<GpuCompilationEnvironment>>
CreateGpuCompEnvFromStringPairs(
    const std::vector<std::pair<std::string, std::string>>& pairs,
    bool strict) {
  std::unique_ptr<GpuCompilationEnvironment> env = CreateDefaultGpuCompEnv();
  for (const auto& [key, value] : pairs) {
    auto* field = GpuCompilationEnvironment::descriptor()->FindFieldByName(key);
    if (field == nullptr) {
      LOG(ERROR) << "Unrecognized flag: '" << key;
      if (strict) {
        return InvalidArgument("Unrecognized flag: %s.", key);
      }
      continue;
    }
    // NOLINTNEXTLINE
    const tsl::protobuf::Reflection* reflection = env->GetReflection();

    if (field->type() == tsl::protobuf::FieldDescriptor::TYPE_STRING) {
      reflection->SetString(env.get(), field, value);
      continue;
    } else if (field->type() == tsl::protobuf::FieldDescriptor::TYPE_INT32) {
      int int_value;
      if (absl::SimpleAtoi(value, &int_value)) {
        reflection->SetInt32(env.get(), field, int_value);
        continue;
      }
    } else if (field->type() == tsl::protobuf::FieldDescriptor::TYPE_INT64) {
      int int_value;
      if (absl::SimpleAtoi(value, &int_value)) {
        reflection->SetInt64(env.get(), field, int_value);
        continue;
      }
    } else if (field->type() == tsl::protobuf::FieldDescriptor::TYPE_FLOAT) {
      float float_value;
      if (absl::SimpleAtof(value, &float_value)) {
        reflection->SetFloat(env.get(), field, float_value);
        continue;
      }
    } else if (field->type() == tsl::protobuf::FieldDescriptor::TYPE_BOOL) {
      bool bvalue = value == "True";
      if (value == "True" || value == "False") {
        reflection->SetBool(env.get(), field, bvalue);
        continue;
      }
    }
    return InvalidArgument(
        "While setting option %s, '%s' is not a valid %s value.", field->name(),
        value, field->type_name());
  }
  return env;
}

namespace {

// Implement a CompilationEnvironment::ProcessNewEnvFn for
// GpuCompilationEnvironment, so that we can add GpuCompilationEnvironments
// to CompilationEnvironments.
//
// The implementation returns Default env if one doesn't exist already.
// NOLINTNEXTLINE
std::unique_ptr<tsl::protobuf::Message> ProcessNewGpuCompilationEnvironment(
    std::unique_ptr<tsl::protobuf::Message> env) {  // NOLINT
  if (!env) {
    return xla::CreateDefaultGpuCompEnv();
  }
  return env;
}

}  // namespace

}  // namespace xla

static bool InitModule() {
  xla::CompilationEnvironments::RegisterProcessNewEnvFn(
      xla::GpuCompilationEnvironment::descriptor(),
      xla::ProcessNewGpuCompilationEnvironment);
  return true;
}
static bool module_initialized = InitModule();
