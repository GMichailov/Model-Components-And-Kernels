from utils import BenchmarkConfig, flush_l2, bytes_per_tensor, max_abs_and_rel_err, run_compile
from collections import defaultdict
from itertools import product
import torch
from pathlib import Path
import torch.nn.functional as F
import json
import statistics


class BenchmarkRMSNormAssumeDivisible:
    def __init__(self, benchmark_config: BenchmarkConfig):
        self.config = benchmark_config
        self.delimiter = "\n\n" + "/" * 50 + "\n\n"
        current_file = Path(__file__).resolve()
        self.target_dir = current_file.parents[1] / "kernels" / "cuda" / "norms"
        device = torch.cuda.current_device()
        self.device_name = torch.cuda.get_device_name(device)
        device_props = torch.cuda.get_device_properties(device)
        self.device_total_memory_bytes = device_props.total_memory
        self.test_cases = {}
        self._create_cases()

    @staticmethod
    def _dtype_size_bytes(dtype: torch.dtype) -> int:
        if dtype in (torch.float16, torch.bfloat16):
            return 2
        if dtype == torch.float32:
            return 4
        raise ValueError(f"Unsupported dtype: {dtype}")

    @staticmethod
    def _dtype_rank(dtype: torch.dtype) -> int:
        """
        Rank by storage/compute precision for your constraint checks.
        fp16 and bf16 are treated as same 'level' here.
        """
        if dtype in (torch.float16, torch.bfloat16):
            return 16
        if dtype == torch.float32:
            return 32
        raise ValueError(f"Unsupported dtype: {dtype}")

    def _create_cases(self):
        """
        Generates plausible RMSNorm benchmark cases.

        Output format:
        {
            (batch_size, seq_len, dim_x): [
                (activation_dtype, weight_dtype, calculation_dtype,
                vectorized_load_count, dim_x, threads_per_block),
                ...
            ],
            ...
        }
        """
        memory_cap_bytes = int(self.device_total_memory_bytes * 0.30)
        VECTOR_LOAD_CAP_BYTES = 16
        cases = defaultdict(list)

        for batch_size, seq_len, dim_x in product(
            self.config.batch_size,
            self.config.seq_len,
            self.config.d_model,
        ):
            for activation_dtype, weight_dtype, calculation_dtype in product(
                self.config.dtypes,
                self.config.dtypes,
                self.config.dtypes,
            ):
                # Disallow mixing fp16 and bf16 anywhere in the same case.
                # Allowed examples:
                #   fp16 / fp16 / fp16
                #   fp16 / fp16 / fp32
                #   bf16 / bf16 / bf16
                #   bf16 / bf16 / fp32
                #   fp32 / fp32 / fp32
                dtype_set = {activation_dtype, weight_dtype, calculation_dtype}
                if torch.float16 in dtype_set and torch.bfloat16 in dtype_set:
                    continue

                # 1) Activation tensor must fit within 30% of total GPU memory
                activation_bytes = (
                    batch_size
                    * seq_len
                    * dim_x
                    * self._dtype_size_bytes(activation_dtype)
                )
                if activation_bytes > memory_cap_bytes:
                    continue

                # 2) activation dtype >= weight dtype
                if self._dtype_rank(activation_dtype) < self._dtype_rank(weight_dtype):
                    continue

                # 3) activation dtype <= calculation dtype
                if self._dtype_rank(activation_dtype) > self._dtype_rank(calculation_dtype):
                    continue

                act_size = self._dtype_size_bytes(activation_dtype)

                for vectorized_load_count in self.config.vectorized_load_sizes:
                    # 4) vectorized load cap
                    if vectorized_load_count * act_size > VECTOR_LOAD_CAP_BYTES:
                        continue

                    # Usually only generate fully divisible vectorized cases
                    if dim_x % vectorized_load_count != 0:
                        continue

                    for threads_per_block in self.config.threads_per_block:
                        # Only keep clean mappings where total per-block coverage divides dim_x
                        if dim_x % (threads_per_block * vectorized_load_count) != 0:
                            continue

                        cases[(batch_size, seq_len, dim_x)].append(
                            (
                                activation_dtype,
                                weight_dtype,
                                calculation_dtype,
                                vectorized_load_count,
                                dim_x,
                                threads_per_block,
                            )
                        )
        self.test_cases = dict(cases)
        self.unique_templates = set(kernel_config for _, kernel_configs in self.test_cases.items() for kernel_config in kernel_configs)
        self._sort_unique_templates()
           
    def _sort_unique_templates(self):
        # Deterministic ordering makes generated files stable
        self.unique_templates = sorted(
            self.unique_templates,
            key=lambda x: (
                str(x[0]),   # activation dtype
                str(x[1]),   # weight dtype
                str(x[2]),   # calculation dtype
                x[3],        # vectorized_load_count
                x[4],        # dim_x
                x[5],        # threads_per_block
            ),
        )

    @staticmethod
    def _torch_dtype_to_cpp(dtype):
        mapping = {
            torch.bfloat16: "at::BFloat16",
            torch.float16: "at::Half",
            torch.float32: "float",
        }
        return mapping[dtype]

    def insert_templates(self):
        TARGET_FILE = self.target_dir / "src" / "rmsnorm.cu"
        append_templates = [self.delimiter]
        for (
            activation_dtype,
            weight_dtype,
            calculation_dtype,
            vectorized_load_count,
            dim_x,
            threads_per_block,
        ) in self.unique_templates:
            activation_cpp = self._torch_dtype_to_cpp(activation_dtype)
            weight_cpp = self._torch_dtype_to_cpp(weight_dtype)
            calculation_cpp = self._torch_dtype_to_cpp(calculation_dtype)
            append_templates.append(
                "template at::Tensor "
                "norms::launchers::offline::cuda_rmsnorm_divisible_forward_launcher_offline"
                f"<{activation_cpp}, {weight_cpp}, {calculation_cpp}, "
                f"{vectorized_load_count}, {dim_x}, {threads_per_block}>"
                "(const at::Tensor&, const at::Tensor&, float);\n"
            )
        append_templates.append(self.delimiter)
        write_str = "".join(append_templates)
        with open(TARGET_FILE, "a", encoding="utf-8") as f:
            f.write(write_str)

    @staticmethod
    def _torch_dtype_to_entry_point_name(dtype):
        mapping = {
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
            torch.float32: "fp32",
        }
        return mapping[dtype]

    def insert_entrypoints(self):
        TARGET_FILE = self.target_dir / "bindings" / "launchers.h"
        append_templates = [self.delimiter]
        self.launchers = []
        # Reorg unique templates
        grouped = defaultdict(lambda: defaultdict(list))
        for (activation_dtype, weight_dtype, calculation_dtype, vectorized_load_count, dim_x, threads_per_block) in self.unique_templates:
            grouped[(activation_dtype, weight_dtype, calculation_dtype, vectorized_load_count)][dim_x].append(threads_per_block)
        grouped_keys = sorted(
            grouped.keys(),
            key=lambda x: (
                str(x[0]),
                str(x[1]),
                str(x[2]),
                x[3],
            ),
        )
        for (act, wgt, calc, vec) in grouped_keys:
            act_name = self._torch_dtype_to_entry_point_name(act)
            wgt_name = self._torch_dtype_to_entry_point_name(wgt)
            calc_name = self._torch_dtype_to_entry_point_name(calc)
            act_cpp = self._torch_dtype_to_cpp(act)
            wgt_cpp = self._torch_dtype_to_cpp(wgt)
            calc_cpp = self._torch_dtype_to_cpp(calc)
            fn_name = f"rmsnorm_forward_divisible_{act_name}_{wgt_name}_{calc_name}_v{vec}"

            entrypoint_base = (
                f"at::Tensor {fn_name}(const at::Tensor& x, const at::Tensor& gamma, float epsilon, int num_threads) {{\n"
                "   switch (x.size(1)) {\n"
            )
            dim_groups = grouped[(act, wgt, calc, vec)]
            for dim_x in sorted(dim_groups.keys()):
                dim_x_base = (
                    f"      case {dim_x}:\n"
                    f"          switch (num_threads) {{\n"
                )
                for threads_per_block in sorted(set(dim_groups[dim_x])):
                    thread_base = (
                        f"              case {threads_per_block}:\n"
                        f"                  return cuda_rmsnorm_divisible_forward_launcher_offline<{act_cpp}, {wgt_cpp}, {calc_cpp}, {vec}, {dim_x}, {threads_per_block}>"
                        f"(x, gamma, epsilon);\n"
                    )
                    dim_x_base += thread_base
                dim_x_base += (
                    f"              default:\n"
                    f"                  TORCH_CHECK(false, \"Unsupported thread count\");\n"
                    f"          }}\n"
                )
                entrypoint_base += dim_x_base
            entrypoint_base += (
                "       default:\n"
                f"          TORCH_CHECK(false, \"Unsupported hidden dim for {fn_name}\");\n"
                "   }\n"
                "}\n\n"
            )
            self.launchers.append(fn_name)
            append_templates.append(entrypoint_base)
        append_templates.append(self.delimiter)
        write_str = "".join(append_templates)
        with open(TARGET_FILE, "a", encoding="utf-8") as f:
            f.write(write_str)

    def insert_bindings(self):
        TARGET_FILE = self.target_dir / "bindings" / "bindings.cpp"
        append_templates = [self.delimiter]
        for fn_name in self.launchers:
            append_templates.append(f"\tm.def(\"{fn_name}\", &{fn_name}, \"x\"_a, \"gamma\"_a, \"epsilon\"_a, \"num_threads\"_a);\n")
        append_templates.append(self.delimiter)
        write_str = "".join(append_templates)
        with open(TARGET_FILE, "r", encoding="utf-8") as f:
            contents = f.read()
        insert_idx = contents.find("\n}")
        contents = contents[:insert_idx] + write_str + contents[insert_idx:]
        with open(TARGET_FILE, "w", encoding="utf-8") as f:
            f.write(contents)

    def run_benchmarks(self):
        def logical_bytes_per_call(x_shape, x_dtype, gamma_shape, gamma_dtype, y_dtype):
            """
            Logical minimum bytes moved per RMSNorm call:
            - read x
            - read gamma
            - write y

            This is best interpreted as EFFECTIVE BANDWIDTH, not true physical HBM traffic.
            """
            x_bytes = bytes_per_tensor(x_shape, x_dtype)
            gamma_bytes = bytes_per_tensor(gamma_shape, gamma_dtype)
            y_bytes = bytes_per_tensor(x_shape, y_dtype)
            return x_bytes + gamma_bytes + y_bytes
        
        @torch.inference_mode()
        def baseline_rmsnorm_eager(x, gamma, eps):
            # Functional baseline: no module mutation, no Parameter cloning/swapping.
            return F.rms_norm(x, (x.shape[-1],), gamma, eps)
        
        torch.set_grad_enabled(False)
        from ComponentsAndKernels.Kernels import fused_norms
        try:
            baseline_rmsnorm_compiled = torch.compile(baseline_rmsnorm_eager, dynamic=False)
        except Exception:
            baseline_rmsnorm_compiled = baseline_rmsnorm_eager
        def benchmark_op(name, op, x_pool, gamma_pool, ref_out, logical_bytes):
            # Correctness
            y = op(x_pool[0], gamma_pool[0], epsilon)
            torch.cuda.synchronize()

            max_abs, max_rel = max_abs_and_rel_err(y, ref_out)
            is_close = torch.allclose(
                y, ref_out,
                atol=self.config.atol,
                rtol=self.config.rtol,
            )

            # Warmup
            for i in range(self.config.warmup_iters):
                idx = i % len(x_pool)
                _ = op(x_pool[idx], gamma_pool[idx], epsilon)
            torch.cuda.synchronize()

            per_repeat_ms = []

            for rep in range(self.config.timing_repeats):
                flush_l2()
                torch.cuda.synchronize()

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                torch.cuda.synchronize()
                start_event.record()

                for i in range(self.config.benchmark_iters):
                    idx = (rep * self.config.benchmark_iters + i) % len(x_pool)
                    _ = op(x_pool[idx], gamma_pool[idx], epsilon)

                end_event.record()
                end_event.synchronize()

                elapsed_ms = start_event.elapsed_time(end_event)
                per_call_ms = elapsed_ms / self.config.benchmark_iters
                per_repeat_ms.append(per_call_ms)

            median_ms = statistics.median(per_repeat_ms)
            mean_ms = statistics.mean(per_repeat_ms)
            min_ms = min(per_repeat_ms)
            max_ms = max(per_repeat_ms)
            stdev_ms = statistics.pstdev(per_repeat_ms) if len(per_repeat_ms) > 1 else 0.0
            effective_bw_gb_s = (logical_bytes / (median_ms / 1000.0)) / 1e9

            return {
                "name": name,
                "median_ms": median_ms,
                "mean_ms": mean_ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "stdev_ms": stdev_ms,
                "effective_bw_gb_s": effective_bw_gb_s,
                "max_abs_err": max_abs,
                "max_rel_err": max_rel,
                "correct": is_close,
            }

        results_by_shape = {}
        best_configs = {}

        epsilon = 1e-5

        print(f"\nRunning RMSNorm benchmarks on: {self.device_name}")
        print(f"Total templates: {len(self.unique_templates)}")
        print(f"Total shapes: {len(self.test_cases)}")

        # Group configs per shape by (act, wgt, calc, vec)
        grouped_cases = defaultdict(list)
        for shape_key, kernel_cfgs in self.test_cases.items():
            grouped_cases[shape_key].extend(kernel_cfgs)

        for (batch_size, seq_len, dim_x), kernel_cfgs in grouped_cases.items():
            print(f"\n{'=' * 120}")
            print(f"Shape: batch={batch_size}, seq_len={seq_len}, dim_x={dim_x}")
            print(f"Candidate kernels: {len(kernel_cfgs)}")

            # Build pools conservatively.
            # This keeps total pool allocation bounded relative to the activation tensor size.
            # You said you're nervous about memory, so this avoids hardcoding an aggressive pool.
            sample_activation_bytes = None
            if kernel_cfgs:
                sample_activation_dtype = kernel_cfgs[0][0]
                x_shape = (batch_size, seq_len, dim_x)
                sample_activation_bytes = bytes_per_tensor(x_shape, sample_activation_dtype)
            else:
                continue

            # Try to keep the pool under ~30% of total VRAM for x only.
            # Gamma is tiny compared to x, so it is ignored for sizing.
            max_pool_bytes = int(self.device_total_memory_bytes * 0.30)
            x_pool_size = max(1, min(8, max_pool_bytes // max(sample_activation_bytes, 1)))

            print(f"x_pool_size={x_pool_size}")

            # Cache pools/baselines per dtype triple so we don't recreate them for every thread count
            per_dtype_bundle = {}

            # Separate baseline result storage by activation/weight/calc combo
            baseline_results = {}

            # Deduplicate by dtype triple because baseline doesn't care about vec/threads
            dtype_triples = sorted(
                set((act, wgt, calc) for act, wgt, calc, _, _, _ in kernel_cfgs),
                key=lambda x: (str(x[0]), str(x[1]), str(x[2]))
            )

            for activation_dtype, weight_dtype, calculation_dtype in dtype_triples:
                x_shape = (batch_size, seq_len, dim_x)
                gamma_shape = (dim_x,)

                try:
                    x_pool = [
                        torch.randn(x_shape, device=self.config.device, dtype=activation_dtype)
                        for _ in range(x_pool_size)
                    ]
                    gamma_pool = [
                        torch.randn(gamma_shape, device=self.config.device, dtype=weight_dtype)
                        for _ in range(x_pool_size)
                    ]
                except RuntimeError as e:
                    print(
                        f"Skipping dtype combo "
                        f"{activation_dtype}, {weight_dtype}, {calculation_dtype} "
                        f"for shape {(batch_size, seq_len, dim_x)} due to OOM during pool creation: {e}"
                    )
                    continue

                with torch.inference_mode():
                    ref_out = F.rms_norm(
                        x_pool[0].to(calculation_dtype),
                        (dim_x,),
                        gamma_pool[0].to(calculation_dtype),
                        epsilon,
                    ).to(activation_dtype)

                logical_bytes = logical_bytes_per_call(
                    x_shape=x_shape,
                    x_dtype=activation_dtype,
                    gamma_shape=gamma_shape,
                    gamma_dtype=weight_dtype,
                    y_dtype=activation_dtype,
                )

                per_dtype_bundle[(activation_dtype, weight_dtype, calculation_dtype)] = {
                    "x_pool": x_pool,
                    "gamma_pool": gamma_pool,
                    "ref_out": ref_out,
                    "logical_bytes": logical_bytes,
                }

                print(
                    f"\nDtypes: act={activation_dtype}, wgt={weight_dtype}, calc={calculation_dtype}"
                    f"\nlogical bytes/call={logical_bytes / 1e6:.3f} MB"
                )

                eager_result = benchmark_op(
                    name="PyTorch eager F.rms_norm",
                    op=baseline_rmsnorm_eager,
                    x_pool=x_pool,
                    gamma_pool=gamma_pool,
                    ref_out=ref_out,
                    logical_bytes=logical_bytes,
                )

                compiled_result = benchmark_op(
                    name="PyTorch compiled F.rms_norm",
                    op=baseline_rmsnorm_compiled,
                    x_pool=x_pool,
                    gamma_pool=gamma_pool,
                    ref_out=ref_out,
                    logical_bytes=logical_bytes,
                )

                baseline_results[(activation_dtype, weight_dtype, calculation_dtype)] = {
                    "eager": eager_result,
                    "compiled": compiled_result,
                }

                print("\nBASELINES")
                print("-" * 120)
                print(
                    f"{'Name':<34} {'Median ms':>10} {'Mean ms':>10} {'Min ms':>10} {'Max ms':>10} "
                    f"{'Std ms':>10} {'Eff BW GB/s':>13} {'Correct':>9} {'MaxAbsErr':>12} {'MaxRelErr':>12}"
                )
                for r in [eager_result, compiled_result]:
                    print(
                        f"{r['name']:<34} "
                        f"{r['median_ms']:>10.4f} "
                        f"{r['mean_ms']:>10.4f} "
                        f"{r['min_ms']:>10.4f} "
                        f"{r['max_ms']:>10.4f} "
                        f"{r['stdev_ms']:>10.4f} "
                        f"{r['effective_bw_gb_s']:>13.2f} "
                        f"{str(r['correct']):>9} "
                        f"{r['max_abs_err']:>12.6f} "
                        f"{r['max_rel_err']:>12.6f}"
                    )

            custom_results = []

            for (
                activation_dtype,
                weight_dtype,
                calculation_dtype,
                vectorized_load_count,
                _dim_x,
                threads_per_block,
            ) in kernel_cfgs:
                dtype_key = (activation_dtype, weight_dtype, calculation_dtype)
                if dtype_key not in per_dtype_bundle:
                    continue

                fn_name = (
                    f"rmsnorm_forward_divisible_"
                    f"{self._torch_dtype_to_entry_point_name(activation_dtype)}_"
                    f"{self._torch_dtype_to_entry_point_name(weight_dtype)}_"
                    f"{self._torch_dtype_to_entry_point_name(calculation_dtype)}_"
                    f"v{vectorized_load_count}"
                )

                if not hasattr(fused_norms, fn_name):
                    print(f"Missing binding: {fn_name}")
                    continue

                func = getattr(fused_norms, fn_name)
                bundle = per_dtype_bundle[dtype_key]
                x_pool = bundle["x_pool"]
                gamma_pool = bundle["gamma_pool"]
                ref_out = bundle["ref_out"]
                logical_bytes = bundle["logical_bytes"]

                def op(x, gamma, eps, _func=func, _threads=threads_per_block):
                    return _func(x, gamma, eps, _threads)

                name = (
                    f"{fn_name} "
                    f"threads={threads_per_block} "
                    f"shape=({batch_size},{seq_len},{dim_x})"
                )

                try:
                    result = benchmark_op(
                        name=name,
                        op=op,
                        x_pool=x_pool,
                        gamma_pool=gamma_pool,
                        ref_out=ref_out,
                        logical_bytes=logical_bytes,
                    )
                    result["batch_size"] = batch_size
                    result["seq_len"] = seq_len
                    result["dim_x"] = dim_x
                    result["activation_dtype"] = str(activation_dtype)
                    result["weight_dtype"] = str(weight_dtype)
                    result["calculation_dtype"] = str(calculation_dtype)
                    result["vec"] = vectorized_load_count
                    result["threads"] = threads_per_block

                    compiled_baseline = baseline_results[dtype_key]["compiled"]
                    result["speedup_vs_compiled"] = (
                        compiled_baseline["median_ms"] / result["median_ms"]
                    )

                    custom_results.append(result)

                except Exception as e:
                    print(f"Skipping {name}: {e}")

            correct_results = [r for r in custom_results if r["correct"]]
            correct_results.sort(key=lambda r: r["median_ms"])

            print("\nTOP 10 FASTEST CORRECT CUSTOM CONFIGURATIONS")
            print("-" * 160)
            print(
                f"{'Rank':<6} {'Vec':<6} {'Threads':<10} {'Median ms':>10} {'Mean ms':>10} "
                f"{'Min ms':>10} {'Max ms':>10} {'Std ms':>10} {'Eff BW GB/s':>13} "
                f"{'vs compiled':>12} {'MaxAbsErr':>12} {'MaxRelErr':>12}"
            )

            for rank, r in enumerate(correct_results[:10], 1):
                print(
                    f"{rank:<6} "
                    f"{r['vec']:<6} "
                    f"{r['threads']:<10} "
                    f"{r['median_ms']:>10.4f} "
                    f"{r['mean_ms']:>10.4f} "
                    f"{r['min_ms']:>10.4f} "
                    f"{r['max_ms']:>10.4f} "
                    f"{r['stdev_ms']:>10.4f} "
                    f"{r['effective_bw_gb_s']:>13.2f} "
                    f"{r['speedup_vs_compiled']:>12.2f}x "
                    f"{r['max_abs_err']:>12.6f} "
                    f"{r['max_rel_err']:>12.6f}"
                )

            bad_results = [r for r in custom_results if not r["correct"]]
            if bad_results:
                print("\nINCORRECT CUSTOM CONFIGURATIONS")
                print("-" * 120)
                for r in bad_results:
                    print(
                        f"{r['name']:<60} "
                        f"median_ms={r['median_ms']:.4f}, "
                        f"max_abs_err={r['max_abs_err']:.6f}, "
                        f"max_rel_err={r['max_rel_err']:.6f}"
                    )

            results_by_shape[(batch_size, seq_len, dim_x)] = {
                "baseline_results": baseline_results,
                "custom_results": custom_results,
                "top_correct": correct_results[:10],
            }

            if correct_results:
                best_configs[(batch_size, seq_len, dim_x)] = correct_results[0]

            # Free pools aggressively between shapes
            del per_dtype_bundle
            torch.cuda.empty_cache()

        # Optional summary artifact
        summary = {
            "device_name": self.device_name,
            "device_total_memory_bytes": self.device_total_memory_bytes,
            "best_configs": {
                str(k): v for k, v in best_configs.items()
            },
        }

        output_path = Path.cwd() / "kernel_configs.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved best config summary to: {output_path}")
        return results_by_shape, best_configs
        
    def remove_changes(self):
        # Remove changes from .cu file
        TARGET_FILE = self.target_dir / "src" / "rmsnorm.cu"
        with open(TARGET_FILE, "r", encoding="utf-8") as f:
            contents = f.read()
        idx = contents.find(self.delimiter)
        if idx != -1:
            new_content = contents[:idx]
            with open(TARGET_FILE, "w", encoding="utf-8") as f:
                f.write(new_content)
        # Remove changes from launchers.h
        TARGET_FILE = self.target_dir / "bindings" / "launchers.h"
        with open(TARGET_FILE, "r", encoding="utf-8") as f:
            contents = f.read()
        idx = contents.find(self.delimiter)
        if idx != -1:
            new_content = contents[:idx]
            with open(TARGET_FILE, "w", encoding="utf-8") as f:
                f.write(new_content)
        # Remove bindings from bindings.cpp by finding in between bindings.
        TARGET_FILE = self.target_dir / "bindings" / "bindings.cpp"
        with open(TARGET_FILE, "r", encoding="utf-8") as f:
            contents = f.read()
        idx = contents.find(self.delimiter)
        if idx != -1:
            end_idx = contents.find(self.delimiter, idx+1) + len(self.delimiter)
            contents = contents[:idx] + contents[end_idx:]
            with open(TARGET_FILE, "w", encoding="utf-8") as f:
                f.write(contents)


if __name__ == "__main__":
    import sys
    benchmark_config = BenchmarkConfig()
    test = BenchmarkRMSNormAssumeDivisible(benchmark_config)
    if sys.argv[1] == "clean":
        test.remove_changes()
    elif sys.argv[1] == "run":
        test.insert_templates()
        test.insert_entrypoints()
        test.insert_bindings()
        run_compile()

    print("Done")