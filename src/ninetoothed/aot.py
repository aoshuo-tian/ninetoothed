import ast
import ctypes
import pathlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import uuid

import ninetoothed.dtype
import ninetoothed.naming as naming
from ninetoothed.generation import CACHE_DIR, CodeGenerator
from ninetoothed.tensor import Tensor
from ninetoothed.utils import calculate_default_configs


def aot(
    func,
    caller="cuda",
    kernel_name=None,
    output_dir=None,
    num_warps=None,
    num_stages=None,
):
    default_num_warps, default_num_stages = calculate_default_configs()

    if num_warps is None:
        num_warps = default_num_warps

    if num_stages is None:
        num_stages = default_num_stages

    output_dir = pathlib.Path(output_dir)

    output_contents = _aot(func, caller, kernel_name, num_warps, num_stages)

    for output_name, output_content in output_contents.items():
        output_path = output_dir / output_name

        with open(output_path, "w") as f:
            f.write(output_content)

    return _generate_launch_func(kernel_name=kernel_name, output_dir=output_dir)


def _aot(func, caller, kernel_name, num_warps, num_stages):
    def _find_tensor_by_source_name(tensors, name):
        name = naming.remove_prefixes(name)

        for tensor in tensors:
            if naming.remove_prefixes(tensor.source.name) == name:
                return tensor

    _HEADER_PATH.parent.mkdir(exist_ok=True)

    if not _HEADER_PATH.exists() or _HEADER_PATH.read_text() != _HEADER_CONTENT:
        _HEADER_PATH.write_text(_HEADER_CONTENT)

    code_generator = CodeGenerator()
    source_file = code_generator(
        func,
        caller=caller,
        kernel_name=kernel_name,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=None,
        prettify=False,
    )

    tensors = code_generator.tensors
    kernel_func = code_generator.kernel_func
    launch_func = code_generator.launch_func

    grid_extractor = _GridExtractor()
    launch_func = grid_extractor.visit(launch_func)
    grid_extractor.visit(code_generator.raw_grid)
    grid = f"{ast.unparse(grid_extractor.grid[0])}, 1, 1"

    launch_arg_names = tuple(arg.arg for arg in launch_func.args.args)
    variant_specs = _enumerate_variant_specs(
        launch_arg_names, tensors, _find_tensor_by_source_name
    )
    _, tensor_ndims, _ = _per_tensor_dim_options(
        launch_arg_names, tensors, _find_tensor_by_source_name
    )

    output_contents = {}

    for (
        variant_suffix,
        divisibility_spec,
        contiguity_spec,
        size_type,
        stride_type,
    ) in variant_specs:
        variant_outputs = _build_variant(
            source_file,
            kernel_func,
            launch_func,
            tensors,
            _find_tensor_by_source_name,
            func,
            kernel_name=kernel_name,
            variant_suffix=variant_suffix,
            grid=grid,
            num_warps=num_warps,
            num_stages=num_stages,
            divisibility_spec=divisibility_spec,
            contiguity_spec=contiguity_spec,
            size_type=size_type,
            stride_type=stride_type,
        )
        output_contents.update(variant_outputs)

    dispatcher_source, dispatcher_header = _generate_dispatcher(
        kernel_name, launch_arg_names, variant_specs, tensor_ndims
    )

    output_contents[f"{kernel_name}.cpp"] = dispatcher_source
    output_contents[f"{kernel_name}.h"] = dispatcher_header

    return output_contents


def _generate_dispatcher(kernel_name, launch_arg_names, variant_specs, tensor_ndims):
    tensor_params = ", ".join(f"NineToothedTensor {name}" for name in launch_arg_names)
    signature_params = (
        f"NineToothedStream stream, {tensor_params}"
        if tensor_params
        else "NineToothedStream stream"
    )
    call_args = (
        f"stream, {', '.join(launch_arg_names)}" if launch_arg_names else "stream"
    )

    signature = f"NineToothedResult launch_{kernel_name}({signature_params})"

    guard = f"NINETOOTHED_{kernel_name.upper()}_H"
    header = (
        f"#ifndef {guard}\n"
        f"#define {guard}\n\n"
        f'#include "{_HEADER_PATH}"\n\n'
        f'#ifdef __cplusplus\nextern "C" {signature};\n'
        f"#else\n{signature};\n#endif\n\n"
        f"#endif\n"
    )

    externs = []
    branches = []

    fallback_call = None

    for (
        variant_suffix,
        divisibility_spec,
        contiguity_spec,
        size_type,
        stride_type,
    ) in variant_specs:
        variant_name = f"launch_{kernel_name}_{variant_suffix}"
        externs.append(
            f'extern "C" NineToothedResult {variant_name}({signature_params});'
        )

        call = f"return {variant_name}({call_args});"

        if (size_type, stride_type) == (
            ninetoothed.dtype.int64,
            ninetoothed.dtype.int64,
        ):
            fallback_call = call

            continue

        checks = tuple(
            f"{name}.shape[{dim}] % 16 == 0" for name, dim in divisibility_spec
        ) + tuple(f"{name}.strides[{dim}] == 1" for name, dim in contiguity_spec)

        if checks:
            branches.append(f"{_INDENTATION}if ({' && '.join(checks)}) {call}")
        else:
            branches.append(f"{_INDENTATION}{call}")

    prelude_lines = []

    if fallback_call is not None and launch_arg_names:
        overflow_terms = _overflow_terms(launch_arg_names, tensor_ndims)

        if overflow_terms:
            prelude_lines.append(
                f"{_INDENTATION}if ({' || '.join(overflow_terms)}) {fallback_call}"
            )

    if fallback_call is None and launch_arg_names:
        overflow_terms = _overflow_terms(launch_arg_names, tensor_ndims)
        if overflow_terms:
            prelude_lines.append(f"{_INDENTATION}if ({' || '.join(overflow_terms)}) return (NineToothedResult)1;")
    body_lines = prelude_lines + branches
    if fallback_call is None and branches:
        body_lines.append(f"{_INDENTATION}return (NineToothedResult)1;")
    if not branches and fallback_call is not None:
        body_lines.append(f"{_INDENTATION}{fallback_call}")

    source = (
        f'#include "{_HEADER_PATH}"\n\n'
        + "\n".join(externs)
        + f'\n\nextern "C" {signature} {{\n'
        + "\n".join(body_lines)
        + "\n}\n"
    )

    return source, header


def _build_variant(
    source_file,
    kernel_func,
    launch_func,
    tensors,
    find_tensor,
    func,
    *,
    kernel_name,
    variant_suffix,
    grid,
    num_warps,
    num_stages,
    divisibility_spec,
    contiguity_spec,
    size_type=ninetoothed.dtype.int32,
    stride_type=ninetoothed.dtype.int32,
):
    divisibility_set = {
        (naming.remove_prefixes(name), dim) for name, dim in divisibility_spec
    }
    contiguity_set = {
        (naming.remove_prefixes(name), dim) for name, dim in contiguity_spec
    }

    param_strings = ["stream"]
    param_types = []
    constexpr_param_indices = []
    constexpr_strides = []

    for arg in kernel_func.args.args:
        param = arg.arg

        param_strings.append(param)

        if match := Tensor.pointer_pattern().fullmatch(param):
            source_name = match.group(1)
            tensor = find_tensor(tensors, source_name)
            dtype = tensor.source.dtype

            param_types.append(f"*{dtype}:16")
        elif match := Tensor.size_pattern().fullmatch(param):
            source_name = match.group(1)
            dim_index = int(match.group(3))
            bare_source_name = naming.remove_prefixes(source_name)

            if (bare_source_name, dim_index) in divisibility_set:
                param_types.append(f"{size_type}:16")
            else:
                param_types.append(size_type)
        elif match := Tensor.stride_pattern().fullmatch(param):
            source_name = match.group(1)
            dim_index = int(match.group(3))
            bare_source_name = naming.remove_prefixes(source_name)

            if (bare_source_name, dim_index) in contiguity_set:
                param_types.append("1")
                constexpr_param_indices.append(len(param_types) - 1)
                constexpr_strides.append((source_name, dim_index))
            else:
                if (os.environ.get("NINETOOTHED_AOT_VARIANTS") == "qkrope_i32"
                    and find_tensor(tensors, source_name).source.ndim != 3):
                    param_types.append(stride_type)
                else:
                    param_types.append(f"{stride_type}:16")
        else:
            source_name = param
            tensor = find_tensor(tensors, source_name)
            dtype = tensor.source.dtype

            if tensor.constexpr:
                param_types.append(f"{tensor.value}")
                constexpr_param_indices.append(len(param_types) - 1)
            else:
                if dtype == ninetoothed.dtype.float32:
                    dtype = ninetoothed.dtype.float64

                param_types.append(dtype)

    signature = ", ".join(param_types)

    for index in sorted(set(constexpr_param_indices), reverse=True):
        param_strings.pop(index + 1)
        param_types.pop(index)

    signature_hash, output_contents = _compile(
        source_file, kernel_name, signature, grid, num_warps, num_stages
    )

    source_candidates = [
        f"{kernel_name}.{signature_hash}{suffix}"
        for suffix in (".c", ".cpp")
        if f"{kernel_name}.{signature_hash}{suffix}" in output_contents
    ]
    if len(source_candidates) != 1:
        raise RuntimeError(
            "Expected exactly one Triton AOT source file; "
            f"found {source_candidates}. "
            f"Available files: {sorted(output_contents)}"
        )
    c_source_file_name = source_candidates[0]
    c_source_file = output_contents[c_source_file_name]

    c_header_file_name = f"{kernel_name}.{signature_hash}.h"
    c_header_file = output_contents[c_header_file_name]

    pattern = rf"\({', '.join(rf'(.*) {param}' for param in param_strings)}\)"
    c_param_type_strings = re.search(pattern, c_header_file).groups()

    kernel_name_with_hash = f"{kernel_name}_{signature_hash}"

    unparser = _Unparser(c_param_type_strings, constexpr_strides)

    launch_func_unparsed = unparser.unparse(launch_func)
    launch_func_unparsed_lines = launch_func_unparsed.splitlines()
    is_hip = "#include <hip/hip_runtime.h>" in c_source_file
    if is_hip:
        c_source_file = _prepare_hip_aot_source(c_source_file)
    # Preserve explicit-context handling for CUDA-compatible drivers.
    context_lines = [
        f"{_INDENTATION}CUcontext nt_context = nullptr;",
        f"{_INDENTATION}CUresult nt_status = cuCtxGetCurrent(&nt_context);",
        f"{_INDENTATION}if (nt_status != CUDA_SUCCESS) return nt_status;",
        f"{_INDENTATION}if (nt_context == nullptr) return CUDA_ERROR_INVALID_CONTEXT;",
        f"{_INDENTATION}unsigned long long ctx_id;",
        f"{_INDENTATION}nt_status = cuCtxGetId(nt_context, &ctx_id);",
        f"{_INDENTATION}if (nt_status != CUDA_SUCCESS) return nt_status;",
    ]
    if is_hip:
        context_lines = [
            f"{_INDENTATION}int nt_device = 0;",
            f"{_INDENTATION}hipError_t nt_status = hipGetDevice(&nt_device);",
            f"{_INDENTATION}if (nt_status != hipSuccess) return nt_status;",
            f"{_INDENTATION}hipCtx_t nt_context = nullptr;",
            f"{_INDENTATION}nt_status = hipCtxGetCurrent(&nt_context);",
            f"{_INDENTATION}if (nt_status != hipSuccess) return nt_status;",
            f"{_INDENTATION}auto ctx_id = std::make_pair("
            "nt_device, reinterpret_cast<std::uintptr_t>(nt_context));",
        ]
    launch_func_unparsed_lines[1:1] = context_lines
    launch_func_unparsed = "\n".join(launch_func_unparsed_lines)
    launch_func_unparsed = launch_func_unparsed.replace(
        func.__name__, f"kernels_{variant_suffix}[ctx_id].{kernel_name_with_hash}"
    )
    launch_func_unparsed = launch_func_unparsed.replace(
        f"launch_{kernel_name}(", f"launch_{kernel_name}_{variant_suffix}(", 1
    )

    c_source_file = c_source_file.replace("<stdint.h>", f'"{_HEADER_PATH}"')
    output_contents[c_source_file_name] = c_source_file

    output_contents.pop(c_header_file_name, None)

    kernel_start = c_source_file.find("//")
    cache_declaration = (
        f"static ninetoothed::ThreadSafeUnorderedMap<unsigned long long, "
        f"{kernel_name_with_hash}::Kernel> kernels_{variant_suffix};"
    )
    if is_hip:
        # Keep includes and error helpers outside the Kernel structure.
        marker = "// globals"
        if c_source_file.count(marker) != 1:
            raise RuntimeError("Unexpected Triton HIP template: globals marker")
        kernel_start = c_source_file.index(marker)
        if "#include" in c_source_file[kernel_start:]:
            raise RuntimeError("Unexpected HIP include inside kernel body")

        from triton.runtime import driver

        target = driver.active.get_current_target()
        if target.backend != "hip":
            raise RuntimeError("HIP template does not match active backend")

        # Use the device target's wave size, not a host-side warpSize symbol.
        c_source_file, replacements = re.subn(
            '\\bwarpSize\\b', str(int(target.warp_size)), c_source_file
        )
        if replacements != 1:
            raise RuntimeError("Unexpected Triton HIP warpSize expression")

        extra_headers = '#include <map>\n#include <utility>\n#include <cstdint>\n'
        c_source_file = extra_headers + c_source_file
        kernel_start += len(extra_headers)
        cache_declaration = (
            "static thread_local std::map<std::pair<int, std::uintptr_t>, "
            f"{kernel_name_with_hash}::Kernel> kernels_{variant_suffix};"
        )
    if kernel_start < 0:
        raise RuntimeError("Cannot locate Triton AOT kernel body")
    kernel_end = len(c_source_file)
    cpp_source_file = (
        c_source_file[:kernel_start]
        + f"namespace {kernel_name_with_hash} {{\n"
        + "struct Kernel {\n"
        + textwrap.indent(c_source_file[kernel_start:kernel_end], _INDENTATION)
        + "};\n"
        + textwrap.indent(c_source_file[kernel_end:], _INDENTATION)
        + "}\n"
        + f"\n{cache_declaration}\n"
        + f'\nextern "C" {launch_func_unparsed}\n'
    )
    cpp_source_file_name = f"{kernel_name}.{variant_suffix}.cpp"
    output_contents[cpp_source_file_name] = cpp_source_file
    output_contents.pop(c_source_file_name)

    return output_contents


def _enumerate_variant_specs(launch_arg_names, tensors, find_tensor):
    if os.environ.get("NINETOOTHED_AOT_VARIANTS") == "embedding_rows_i32":
        _, ndims, _ = _per_tensor_dim_options(launch_arg_names, tensors, find_tensor)
        if tuple(ndims) != (2, 1, 2):
            raise ValueError("embedding_rows_i32 requires output/index/weight ranks2/1/2")
        # Row count is intentionally dynamic. Only the feature sizes are :16.
        division = tuple((name, 1) for name, ndim in zip(launch_arg_names, ndims) if ndim == 2)
        contiguous = tuple((name, ndim - 1) for name, ndim in zip(launch_arg_names, ndims))
        suffix = _variant_suffix(division, contiguous, launch_arg_names, ndims,
                                 ninetoothed.dtype.int32, ninetoothed.dtype.int32)
        return [(suffix, division, contiguous, ninetoothed.dtype.int32, ninetoothed.dtype.int32)]
    if os.environ.get("NINETOOTHED_AOT_VARIANTS") == "qkrope_i32":
        _, ndims, _ = _per_tensor_dim_options(launch_arg_names, tensors, find_tensor)
        division = tuple((name, 2) for name, ndim in zip(launch_arg_names, ndims) if ndim == 3)
        contiguous = tuple((name, 2) for name, ndim in zip(launch_arg_names, ndims) if ndim == 3)
        contiguous += ((launch_arg_names[6], 1),)  # Actual cache inner stride, no position-stride assumptions.
        suffix = _variant_suffix(division, contiguous, launch_arg_names, ndims,
                                 ninetoothed.dtype.int32, ninetoothed.dtype.int32)
        return [(suffix, division, contiguous, ninetoothed.dtype.int32, ninetoothed.dtype.int32)]
    per_tensor_dims, tensor_ndims, innermost_dims = _per_tensor_dim_options(
        launch_arg_names, tensors, find_tensor
    )

    def _spec_from_combo(combo):
        return tuple(
            (name, dim) for name, dim in zip(launch_arg_names, combo) if dim is not None
        )

    base_combo = tuple(dims[0] for dims in per_tensor_dims)
    combos = [base_combo] if any(dim is not None for dim in base_combo) else []

    for i, dims in enumerate(per_tensor_dims):
        if len(dims) <= 1:
            continue

        for alternative_dim in dims[1:]:
            if alternative_dim is None:
                continue

            combo = list(base_combo)
            combo[i] = alternative_dim
            combos.append(tuple(combo))

    dim_specs = tuple(_spec_from_combo(combo) for combo in combos) + ((),)

    specs = []

    for divisibility_spec in dim_specs:
        for contiguity_spec in dim_specs:
            suffix = _variant_suffix(
                divisibility_spec,
                contiguity_spec,
                launch_arg_names,
                tensor_ndims,
                size_type=ninetoothed.dtype.int32,
                stride_type=ninetoothed.dtype.int32,
            )
            specs.append(
                (
                    suffix,
                    divisibility_spec,
                    contiguity_spec,
                    ninetoothed.dtype.int32,
                    ninetoothed.dtype.int32,
                )
            )

    def _num_innermost(spec):
        return sum(1 for name, dim in spec if innermost_dims.get(name) == dim)

    def _specificity(entry):
        _, divisibility_spec, contiguity_spec, _, _ = entry

        return (
            -len(divisibility_spec),
            -_num_innermost(divisibility_spec),
            -len(contiguity_spec),
            -_num_innermost(contiguity_spec),
        )

    specs.sort(key=_specificity)

    fallback_suffix = _variant_suffix(
        (),
        (),
        launch_arg_names,
        tensor_ndims,
        size_type=ninetoothed.dtype.int64,
        stride_type=ninetoothed.dtype.int64,
    )
    specs.append(
        (
            fallback_suffix,
            (),
            (),
            ninetoothed.dtype.int64,
            ninetoothed.dtype.int64,
        )
    )

    # Generic int64 addressing is useful for bounded offline bring-up builds.
    # The default retains every original specialization and its dispatch order.
    policy = os.environ.get("NINETOOTHED_AOT_VARIANTS", "all")
    if policy == "generic":
        return specs[-1:]
    if policy in ("inner32", "inner16"):
        # Strict entry points only: caller proves unit inner strides and int32
        # metadata/offset bounds. No generic fallback is emitted for this policy.
        inner_spec = tuple((name, dim) for name, dim in innermost_dims.items())
        selected = [entry for entry in specs
                    if entry[1] == (inner_spec if policy == "inner16" else ()) and entry[2] == inner_spec
                    and entry[3] == ninetoothed.dtype.int32]
        if len(selected) != 1:
            raise RuntimeError("Cannot select the strict inner32 AOT variant")
        return selected
    if policy != "all":
        raise ValueError("NINETOOTHED_AOT_VARIANTS must be all, generic, inner32 or inner16")
    return specs


def _per_tensor_dim_options(launch_arg_names, tensors, find_tensor):
    per_tensor_dims = []
    tensor_ndims = []

    for name in launch_arg_names:
        tensor = find_tensor(tensors, name)
        ndim = tensor.source.ndim if tensor is not None else 0
        tensor_ndims.append(ndim)

        if ndim == 0:
            per_tensor_dims.append((None,))
        elif ndim == 1:
            per_tensor_dims.append((0,))
        else:
            per_tensor_dims.append((ndim - 1, ndim - 2))

    per_tensor_dims = tuple(per_tensor_dims)
    tensor_ndims = tuple(tensor_ndims)

    innermost_dims = {
        name: dims[0]
        for name, dims in zip(launch_arg_names, per_tensor_dims)
        if dims[0] is not None
    }

    return per_tensor_dims, tensor_ndims, innermost_dims


def _overflow_terms(launch_arg_names, tensor_ndims):
    int32_min = -(2**31)
    int32_max = 2**31 - 1

    return tuple(
        term
        for name, ndim in zip(launch_arg_names, tensor_ndims)
        for dim in range(ndim)
        for term in (
            f"{name}.shape[{dim}] > {int32_max}ULL",
            f"{name}.strides[{dim}] > {int32_max}LL",
            f"{name}.strides[{dim}] < {int32_min}LL",
        )
    )


def _variant_suffix(
    divisibility_spec,
    contiguity_spec,
    launch_arg_names,
    tensor_ndims,
    size_type=ninetoothed.dtype.int32,
    stride_type=ninetoothed.dtype.int32,
):
    divisibility_part = _divisibility_suffix(
        divisibility_spec, launch_arg_names, tensor_ndims
    )
    contiguity_part = _contiguity_suffix(
        contiguity_spec, launch_arg_names, tensor_ndims
    )

    return (
        f"{divisibility_part}_{contiguity_part}_size_{size_type}_stride_{stride_type}"
    )


def _divisibility_suffix(divisibility_spec, launch_arg_names, tensor_ndims):
    hinted = set(divisibility_spec)

    parts = tuple(
        "16" if (name, dim) in hinted else "1"
        for name, ndim in zip(launch_arg_names, tensor_ndims)
        for dim in range(ndim)
    )

    return "divisibility_" + "_".join(parts) if parts else "divisibility"


def _contiguity_suffix(contiguity_spec, launch_arg_names, tensor_ndims):
    contiguous = set(contiguity_spec)

    parts = tuple(
        "1" if (name, dim) in contiguous else "0"
        for name, ndim in zip(launch_arg_names, tensor_ndims)
        for dim in range(ndim)
    )

    return "contiguity_" + "_".join(parts) if parts else "contiguity"


_INDENTATION = "    "

_MACRO_MAPPING = {
    True: ("NINETOOTHED_TRUE", 1),
    False: ("NINETOOTHED_FALSE", 0),
    None: ("NINETOOTHED_NONE", 0),
}

_DTYPE_MAPPING = {
    ninetoothed.dtype.int8: "NINETOOTHED_INT8",
    ninetoothed.dtype.int16: "NINETOOTHED_INT16",
    ninetoothed.dtype.int32: "NINETOOTHED_INT32",
    ninetoothed.dtype.int64: "NINETOOTHED_INT64",
    ninetoothed.dtype.uint8: "NINETOOTHED_UINT8",
    ninetoothed.dtype.uint16: "NINETOOTHED_UINT16",
    ninetoothed.dtype.uint32: "NINETOOTHED_UINT32",
    ninetoothed.dtype.uint64: "NINETOOTHED_UINT64",
    ninetoothed.dtype.float16: "NINETOOTHED_FLOAT16",
    ninetoothed.dtype.bfloat16: "NINETOOTHED_BFLOAT16",
    ninetoothed.dtype.float32: "NINETOOTHED_FLOAT32",
    ninetoothed.dtype.float64: "NINETOOTHED_FLOAT64",
}

_DTYPE_TO_INDEX = {name: i for i, name in enumerate(_DTYPE_MAPPING.keys())}

_MACRO_CONTENT = "\n\n".join(
    f"#define {identifier} {replacement}"
    for identifier, replacement in _MACRO_MAPPING.values()
)

_DATA_TYPE_BODY_CONTENT = ",\n    ".join(_DTYPE_MAPPING.values())

_TEMPLATES_DIR = pathlib.Path(__file__).parent / "templates"

_AUTO_TUNING_CACHE_CONTENT = (
    (_TEMPLATES_DIR / "auto_tuning_cache.h").read_text().strip()
)

_THREAD_SAFE_UNORDERED_MAP_CONTENT = (
    (_TEMPLATES_DIR / "thread_safe_unordered_map.h").read_text().strip()
)

_HEADER_CONTENT = f"""#ifndef NINETOOTHED_H
#define NINETOOTHED_H

#include <stdint.h>

{_MACRO_CONTENT}

enum NineToothedDataType {{
    {_DATA_TYPE_BODY_CONTENT}
}};

typedef struct {{
    void *data;
    uint64_t *shape;
    int64_t *strides;
}} NineToothedTensor;

typedef void *NineToothedStream;

typedef int NineToothedResult;

#ifdef __cplusplus
{_AUTO_TUNING_CACHE_CONTENT}

{_THREAD_SAFE_UNORDERED_MAP_CONTENT}
#endif

#endif // NINETOOTHED_H
"""

_HEADER_PATH = CACHE_DIR / "ninetoothed.h"


class _Unparser:
    def __init__(self, param_types, constexpr_inner_strides=()):
        self._param_types = param_types

        self._constexpr_inner_strides = set(constexpr_inner_strides)

    def unparse(self, node):
        method_name = "_unparse_" + node.__class__.__name__

        if hasattr(self, method_name):
            return getattr(self, method_name)(node)

        return self._generic_unparse(node)

    def _generic_unparse(self, node):
        return ast.unparse(node)

    def _unparse_Expr(self, node):
        return self.unparse(node.value)

    def _unparse_Call(self, node):
        call = ast.Call(
            func=node.func,
            args=[ast.Name(id="stream", ctx=ast.Load())]
            + [arg for arg in node.args if not self._is_excluded(arg)],
            keywords=[],
        )

        unparsed = f"return {self._generic_unparse(call)};"

        pattern = rf"\((stream), {', '.join(r'([^,]*)' for _ in range(len(self._param_types) - 1))}\)"
        args = re.search(pattern, unparsed).groups()

        for i, (arg, type) in enumerate(zip(args, self._param_types)):
            if i != 0 and "." not in arg:
                new_arg = f"*({type} *){arg}.data"
            else:
                new_arg = f"({type}){arg}"

            unparsed = unparsed.replace(arg, new_arg)

        return unparsed

    def _unparse_FunctionDef(self, node):
        params = ["NineToothedStream stream"]
        params += [f"NineToothedTensor {arg.arg}" for arg in node.args.args]
        header = f"NineToothedResult {node.name}({', '.join(params)})"

        self.header = header

        body_lines = []

        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                continue

            stmt_unparsed = self.unparse(stmt)

            if isinstance(stmt, ast.Expr):
                stmt_unparsed = stmt_unparsed.strip()

            if not stmt_unparsed.endswith(";"):
                stmt_unparsed += ";"

            body_lines.append("    " + stmt_unparsed)

        body = "\n".join(body_lines)

        return f"{header} {{\n{body}\n}}"

    def _is_excluded(self, arg):
        if isinstance(arg, ast.Name) and naming.is_constexpr(arg.id):
            return True

        if (
            isinstance(arg, ast.Subscript)
            and isinstance(arg.value, ast.Attribute)
            and arg.value.attr == "strides"
            and isinstance(arg.slice, ast.Constant)
            and isinstance(arg.value.value, ast.Name)
        ):
            return (
                arg.value.value.id,
                arg.slice.value,
            ) in self._constexpr_inner_strides

        return False


class _GridExtractor(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)

        if isinstance(node.op, ast.FloorDiv):
            node.op = ast.Div()

        return node

    def visit_Call(self, node):
        self.generic_visit(node)

        node.func = node.func.value

        return node

    def visit_Lambda(self, node):
        self.generic_visit(node)

        self.grid = node.body.elts

        return node


class _ArgumentTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.POINTER(ctypes.c_uint64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
    ]

    @staticmethod
    def from_torch_tensor(tensor):
        ndim = tensor.ndim
        shape_array_type = _SHAPE_ARRAY_TYPES_BY_NDIM[ndim]
        strides_array_type = _STRIDES_ARRAY_TYPES_BY_NDIM[ndim]

        shape = shape_array_type(*tensor.shape)
        strides = strides_array_type(*tensor.stride())

        arg_tensor = _ArgumentTensor(tensor.data_ptr(), shape, strides)
        arg_tensor._torch_tensor = tensor

        return arg_tensor

    @staticmethod
    def from_scalar(value, ctype):
        buffer = ctype(value)
        arg_tensor = _ArgumentTensor(
            ctypes.addressof(buffer), _EMPTY_SHAPE_ARRAY, _EMPTY_STRIDES_ARRAY
        )
        arg_tensor._buffer = buffer

        return arg_tensor


_MAX_NUM_DIMS = 8

_SHAPE_ARRAY_TYPES_BY_NDIM = tuple(ctypes.c_uint64 * i for i in range(_MAX_NUM_DIMS))

_STRIDES_ARRAY_TYPES_BY_NDIM = tuple(ctypes.c_int64 * i for i in range(_MAX_NUM_DIMS))

_EMPTY_SHAPE_ARRAY = _SHAPE_ARRAY_TYPES_BY_NDIM[0]()

_EMPTY_STRIDES_ARRAY = _STRIDES_ARRAY_TYPES_BY_NDIM[0]()


class _KernelLaunchError(RuntimeError):
    def __init__(self, error_code):
        self._message = f"Kernel launch failed with error code: {error_code}."

        super().__init__(self._message)


def _compile(path, name, signature, grid, num_warps, num_stages):
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = pathlib.Path(temp_dir)
        output_name = uuid.uuid4().hex
        output_path = output_dir / output_name

        command = [
            sys.executable,
            "-m",
            "triton.tools.compile",
            str(path),
            "--kernel-name",
            str(name),
            "--signature",
            str(signature),
            "--grid",
            str(grid),
            "--num-warps",
            str(num_warps),
            "--num-stages",
            str(num_stages),
            "--out-path",
            str(output_path),
        ]

        subprocess.run(command, check=True)

        matching_files = list(output_dir.glob(f"{output_name}.*"))

        signature_hash = matching_files[0].name.split(".")[1]

        output_contents = {}

        for file in matching_files:
            with file.open() as f:
                output_contents[file.name.replace(output_name, name)] = f.read()

    return signature_hash, output_contents


def _generate_launch_func(kernel_name, output_dir):
    output_dir = pathlib.Path(output_dir)

    _compile_library(kernel_name, output_dir)

    return _load_launch_func(kernel_name, output_dir)


def _load_launch_func(kernel_name, output_dir):
    import torch

    library = _load_library(kernel_name, output_dir)
    launch_func_name = f"launch_{kernel_name}"
    launch_func = getattr(library, launch_func_name)

    def _num_tensor_args(kernel_name, output_dir):
        header_path = pathlib.Path(output_dir) / f"{kernel_name}.h"
        text = header_path.read_text()
        pattern = (
            rf"NineToothedResult\s+launch_{re.escape(kernel_name)}\s*\((.*?)\)\s*;"
        )
        match = re.search(pattern, text)

        if match is None:
            raise RuntimeError(
                f"Could not find launch signature for `{kernel_name}` in `{header_path}`."
            )

        params = (param.strip() for param in match.group(1).split(","))

        return sum(1 for param in params if param.startswith("NineToothedTensor "))

    num_tensor_args = _num_tensor_args(kernel_name, output_dir)

    dtype_to_index = _DTYPE_TO_INDEX
    from_torch_tensor = _ArgumentTensor.from_torch_tensor
    from_scalar = _ArgumentTensor.from_scalar
    c_double = ctypes.c_double
    c_int64 = ctypes.c_int64
    c_void_p = ctypes.c_void_p
    current_device = torch.cuda.current_device
    get_current_raw_stream = torch._C._cuda_getCurrentRawStream
    Tensor_cls = torch.Tensor

    def _run_launch_func(*args):
        arguments = [None] * len(args)

        for i, arg in enumerate(args):
            if i < num_tensor_args:
                if isinstance(arg, Tensor_cls):
                    arguments[i] = from_torch_tensor(arg)
                elif type(arg) is float:
                    arguments[i] = from_scalar(arg, c_double)
                elif type(arg) is int:
                    arguments[i] = from_scalar(arg, c_int64)
                else:
                    arguments[i] = arg
            elif type(arg) is str:
                arguments[i] = dtype_to_index[arg]
            else:
                arguments[i] = arg

        stream = c_void_p(get_current_raw_stream(current_device()))
        result = launch_func(stream, *arguments)

        if result != 0:
            raise _KernelLaunchError(result)

    return _run_launch_func



def _prepare_hip_aot_source(source):
    old = "    const char *str;\n    hipDrvGetErrorString(code, &str);"
    new = (
        "    const char *str = nullptr;\n"
        "    hipError_t error_status = hipDrvGetErrorString(code, &str);\n"
        "    if (error_status != hipSuccess || str == nullptr) {\n"
        '        str = "HIP error text unavailable";\n'
        "    }"
    )
    if source.count(old) != 1:
        raise RuntimeError("Unexpected HIP error-string helper")
    source = source.replace(old, new, 1)

    pattern = (
        r"(\bvoid (?:load|unload)_[A-Za-z0-9_]+\([^)]*\)\s*\{)"
        r"([^{}]*)(\})"
    )

    def add_return(match):
        return match[1] + match[2] + "    return;\n" + match[3]

    source, count = re.subn(pattern, add_return, source)
    if count != 2:
        raise RuntimeError("Unexpected HIP load/unload function structure")
    return source


def _compile_library(kernel_name, output_dir):
    source_files = sorted(output_dir.glob(f"{kernel_name}*.cpp"))
    if not source_files:
        raise RuntimeError("No NineToothed AOT source files to compile")

    is_hip = any(
        "#include <hip/hip_runtime.h>" in path.read_text()
        for path in source_files
    )
    if is_hip:
        candidates = (
            "/opt/dtk/hip/bin/hipcc",
            "/opt/dtk/bin/hipcc",
            "hipcc",
        )
        compiler = next(
            (resolved for candidate in candidates
             if (resolved := shutil.which(candidate))),
            None,
        )
        if compiler is None:
            raise RuntimeError("HIP AOT requires hipcc; compiler not found")
        command = [
            compiler,
            "--offload-host-only",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-Wl,-z,defs",
            "-o",
            str(output_dir / f"{kernel_name}.so"),
        ] + [str(path) for path in source_files]
        print("NineToothed HIP host compile:", command, flush=True)
        subprocess.run(command, check=True)
        return

    command = [
        "nvcc",
        "-shared",
        "-arch",
        "native",
        "--threads",
        "0",
        "-Xcompiler",
        "-fPIC",
        # TODO: Remove the following 2 lines after the return value issue is resolved.
        "-Xcompiler",
        "-Wno-return-type",
        "-lcuda",
        "-o",
        output_dir / f"{kernel_name}.so",
    ] + list(output_dir.glob(f"{kernel_name}*.cpp"))

    subprocess.run(command, check=True)


def _load_library(kernel_name, kernel_dir):
    suffix = ".so"

    original_path = kernel_dir / f"{kernel_name}{suffix}"

    with tempfile.NamedTemporaryFile(suffix=suffix) as temp_file:
        temp_path = temp_file.name

        shutil.copy(original_path, temp_path)

        library = ctypes.CDLL(temp_path)

    return library
