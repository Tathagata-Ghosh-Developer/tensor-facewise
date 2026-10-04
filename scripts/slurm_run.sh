#!/usr/bin/env bash
#SBATCH --job-name=tensor_run
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --nodes=1-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --output=run_%j.out
#SBATCH --error=run_%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND} (exit=${rc})" >&2; exit ${rc}' ERR

echo "[INFO] job_id=${SLURM_JOB_ID:-na} host=$(hostname) submit_dir=${SLURM_SUBMIT_DIR:-na} pwd=$PWD"
echo "[INFO] Running scalability matrix: serial, openmp, cuda, hybrid"

ROOT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
if [ ! -f "$ROOT_DIR/Makefile" ]; then
    ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
fi
cd "$ROOT_DIR"
echo "[INFO] root_dir=$ROOT_DIR"

CPU_BIN="$ROOT_DIR/tensor_app_cpu"
CUDA_BIN="$ROOT_DIR/tensor_app_cuda"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/modules.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
    elif [ -f /usr/share/Modules/init/bash ]; then
        # shellcheck disable=SC1091
        source /usr/share/Modules/init/bash
    fi
fi

if command -v module >/dev/null 2>&1; then
    module load gcc/8.5.0 >/dev/null 2>&1 || true
    module load cuda/12.6 >/dev/null 2>&1 || true
fi

prepend_ld_library_path() {
    local path_to_add="$1"
    if [ ! -d "$path_to_add" ]; then
        return
    fi
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${path_to_add}:"*)
            ;;
        *)
            export LD_LIBRARY_PATH="${path_to_add}:${LD_LIBRARY_PATH:-}"
            ;;
    esac
}

discover_cuda_runtime_dirs() {
    local candidate_dirs=(
        "${CUDA_HOME:-}/lib64"
        "${CUDA_HOME:-}/targets/x86_64-linux/lib"
        /usr/local/cuda/lib64
        /usr/local/cuda/targets/x86_64-linux/lib
        /usr/local/cuda-12.6/lib64
        /usr/local/cuda-12.6/targets/x86_64-linux/lib
        /usr/lib64/nvidia
        /opt/cuda/lib64
    )

    if command -v nvcc >/dev/null 2>&1; then
        local nvcc_path nvcc_root
        nvcc_path=$(command -v nvcc)
        if command -v readlink >/dev/null 2>&1; then
            nvcc_path=$(readlink -f "$nvcc_path" 2>/dev/null || echo "$nvcc_path")
        fi
        nvcc_root=$(cd "$(dirname "$nvcc_path")/.." && pwd)
        candidate_dirs+=("$nvcc_root/lib64" "$nvcc_root/lib" "$nvcc_root/targets/x86_64-linux/lib")
    fi

    if command -v ldconfig >/dev/null 2>&1; then
        local libcudart_path
        libcudart_path=$(ldconfig -p 2>/dev/null | awk '/libcudart\.so(\.|$)/ {print $NF; exit}' || true)
        if [ -n "$libcudart_path" ]; then
            candidate_dirs+=("$(dirname "$libcudart_path")")
        fi
    fi

    local cuda_dir
    for cuda_dir in "${candidate_dirs[@]}"; do
        prepend_ld_library_path "$cuda_dir"
    done
}

search_cuda_runtime_dirs() {
    local search_roots=(
        "${CUDA_HOME:-}"
        /usr/local/cuda
        /usr/local
        /opt
        /apps
        /sw
        /cm/shared
    )

    local root
    for root in "${search_roots[@]}"; do
        [ -n "$root" ] || continue
        [ -d "$root" ] || continue

        while IFS= read -r found_lib; do
            [ -n "$found_lib" ] || continue
            prepend_ld_library_path "$(dirname "$found_lib")"
        done < <(find "$root" -maxdepth 7 -type f \( -name 'libcudart.so.12' -o -name 'libcublas.so.12' \) 2>/dev/null | head -n 48 || true)
    done
}

missing_cuda_libs() {
    local cuda_bin="$1"
    ldd "$cuda_bin" 2>/dev/null | awk '/not found/ {print $1}' || true
}

ensure_cuda_binary_runtime() {
    local cuda_bin="$1"
    if ! command -v ldd >/dev/null 2>&1; then
        return
    fi

    local missing_libs
    missing_libs=$(missing_cuda_libs "$cuda_bin")

    if [ -n "$missing_libs" ]; then
        search_cuda_runtime_dirs
        missing_libs=$(missing_cuda_libs "$cuda_bin")
    fi

    if [ -n "$missing_libs" ] && command -v nvcc >/dev/null 2>&1; then
        local nvcc_path nvcc_root
        nvcc_path=$(command -v nvcc)
        if command -v readlink >/dev/null 2>&1; then
            nvcc_path=$(readlink -f "$nvcc_path" 2>/dev/null || echo "$nvcc_path")
        fi
        nvcc_root=$(cd "$(dirname "$nvcc_path")/.." && pwd)

        local lib_name found_path
        while IFS= read -r lib_name; do
            [ -z "$lib_name" ] && continue
            found_path=$(find "$nvcc_root" -type f -name "$lib_name" -print -quit 2>/dev/null || true)
            if [ -n "$found_path" ]; then
                prepend_ld_library_path "$(dirname "$found_path")"
            fi
        done <<< "$missing_libs"
    fi

    missing_libs=$(missing_cuda_libs "$cuda_bin")
    if [ -n "$missing_libs" ]; then
        echo "[ERROR] CUDA runtime libraries unresolved for $cuda_bin" >&2
        echo "[ERROR] Missing libraries:" >&2
        echo "$missing_libs" >&2
        echo "[INFO] CUDA_HOME=${CUDA_HOME:-unset}" >&2
        echo "[INFO] nvcc=$(command -v nvcc 2>/dev/null || echo missing)" >&2
        echo "[INFO] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" >&2
        ldd "$cuda_bin" >&2 || true
        exit 1
    fi
}

make -C "$ROOT_DIR" cpu
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[ERROR] nvcc not found in PATH; CUDA mode is required for this script." >&2
    exit 1
fi
make -C "$ROOT_DIR" cuda

if [ ! -x "$CUDA_BIN" ]; then
    echo "[ERROR] CUDA binary missing or not executable: $CUDA_BIN" >&2
    exit 1
fi
discover_cuda_runtime_dirs
ensure_cuda_binary_runtime "$CUDA_BIN"
CUDA_RUNNER="$CUDA_BIN"

ALLOC_THREADS=${SLURM_CPUS_PER_TASK:-36}
THREAD_CANDIDATES=(1 2 4 8 16 32 36 64 72)
THREAD_LIST=("${THREAD_CANDIDATES[@]}")

echo "[INFO] thread_sweep=${THREAD_LIST[*]} (allocated_threads=${ALLOC_THREADS}; values above allocation are intentional oversubscription points)"

SPARSITY=0.99
DENSITY=$(awk -v s="$SPARSITY" 'BEGIN{printf "%.6f", 1.0 - s}')
WARMUP=1
GPU_RATIO=0.80
CHUNK_AUTO=0

ROWS_SERIES=(100 250 500 1000 2000 3500 5000)
DEPTH_SERIES=(3 5 10 20 40 70 100)

if [ "${#ROWS_SERIES[@]}" -ne "${#DEPTH_SERIES[@]}" ]; then
    echo "[ERROR] ROWS_SERIES and DEPTH_SERIES lengths differ" >&2
    exit 1
fi

OUTPUT_DIR="$ROOT_DIR/artifacts/scalability"
mkdir -p "$OUTPUT_DIR"

RESULT_CSV="$OUTPUT_DIR/scalability_matrix.csv"
LOAD_CSV="$OUTPUT_DIR/load_imbalance.csv"

echo "mode,threads,thread_oversub,shape_class,tensor_shape,input_shape_a,input_shape_b,depth,storage_size_input,storage_size_output,sparsity,omp_schedule,omp_chunk,input_mem_mib,working_set_mem_mib,dense_equiv_working_set_mem_mib,time_sec,iterations,time_per_iteration,speedup_vs_serial,gflops,dense_slices,sparse_slices,dense_sec,sparse_sec,work_units_total,work_units_dense,work_units_sparse,max_thread_work,min_thread_work,thread_work_imbalance,max_thread_time,min_thread_time,thread_time_imbalance,gpu_assigned_slices,cpu_assigned_slices,gpu_assigned_work,cpu_assigned_work,gpu_assigned_ratio,partition_load_imbalance,input_slices_ell,input_slices_coo,input_dense_mem_mib,input_coo_mem_mib,input_ell_mem_mib,input_active_mem_mib,input_compression_dense_to_coo,input_compression_dense_to_ell,input_compression_dense_to_active,input_ell_slot_utilization" > "$RESULT_CSV"
echo "mode,shape_class,tensor_shape,input_shape_a,input_shape_b,depth,storage_size_input,sparsity,working_set_mem_mib,min_time_sec,max_time_sec,sweep_load_imbalance,best_threads,worst_threads,best_thread_work_imbalance,worst_thread_work_imbalance,best_thread_time_imbalance,worst_thread_time_imbalance" > "$LOAD_CSV"

export OMP_SCHEDULE=dynamic

extract_time() {
    local log_file="$1"
    local line
    line=$(grep -E "operation=hadamard time_sec=" "$log_file" | tail -n 1 || true)
    if [ -z "$line" ]; then
        echo ""
        return
    fi
    echo "$line" | sed -E 's/.*time_sec=([0-9.eE+-]+).*/\1/'
}

extract_metric() {
    local key="$1"
    local log_file="$2"
    local line
    line=$(grep -E "^metrics " "$log_file" | tail -n 1 || true)
    if [ -z "$line" ]; then
        echo ""
        return
    fi
    echo "$line" | sed -E "s/.*${key}=([0-9.eE+-]+).*/\\1/"
}

extract_metric_default() {
    local key="$1"
    local log_file="$2"
    local default_value="$3"
    local value
    value=$(extract_metric "$key" "$log_file")
    if [ -z "$value" ]; then
        echo "$default_value"
    else
        echo "$value"
    fi
}

extract_storage_metric() {
    local key="$1"
    local log_file="$2"
    local line
    line=$(grep -E "^storage " "$log_file" | tail -n 1 || true)
    if [ -z "$line" ]; then
        echo ""
        return
    fi
    echo "$line" | sed -E "s/.*${key}=([0-9.eE+-]+).*/\\1/"
}

extract_storage_metric_default() {
    local key="$1"
    local log_file="$2"
    local default_value="$3"
    local value
    value=$(extract_storage_metric "$key" "$log_file")
    if [ -z "$value" ]; then
        echo "$default_value"
    else
        echo "$value"
    fi
}

calc_ratio() {
    local num="$1"
    local den="$2"
    awk -v n="$num" -v d="$den" 'BEGIN{if (d == 0) {print "0"} else {printf "%.10f", n / d}}'
}

calc_gflops() {
    local flops="$1"
    local time_sec="$2"
    awk -v f="$flops" -v t="$time_sec" 'BEGIN{if (t == 0) {print "0"} else {printf "%.10f", (f / t) / 1e9}}'
}

calc_load_imbalance() {
    local max_t="$1"
    local min_t="$2"
    awk -v mx="$max_t" -v mn="$min_t" 'BEGIN{if (mn == 0) {print "0"} else {printf "%.10f", (mx - mn) / mn}}'
}

is_less() {
    local lhs="$1"
    local rhs="$2"
    awk -v a="$lhs" -v b="$rhs" 'BEGIN{exit (a < b) ? 0 : 1}'
}

is_greater() {
    local lhs="$1"
    local rhs="$2"
    awk -v a="$lhs" -v b="$rhs" 'BEGIN{exit (a > b) ? 0 : 1}'
}

storage_size_of() {
    local rows="$1"
    local cols="$2"
    local depth="$3"
    awk -v r="$rows" -v c="$cols" -v d="$depth" 'BEGIN{printf "%.0f", r * c * d}'
}

storage_size_pair_of() {
    local rows_a="$1"
    local cols_a="$2"
    local rows_b="$3"
    local cols_b="$4"
    local depth="$5"
    local a_entries b_entries
    a_entries=$(storage_size_of "$rows_a" "$cols_a" "$depth")
    b_entries=$(storage_size_of "$rows_b" "$cols_b" "$depth")
    awk -v a="$a_entries" -v b="$b_entries" 'BEGIN{printf "%.0f", a + b}'
}

estimate_nnz() {
    local entries="$1"
    local density="$2"
    awk -v e="$entries" -v d="$density" 'BEGIN{printf "%.0f", e * d}'
}

bytes_to_mib() {
    local bytes="$1"
    awk -v b="$bytes" 'BEGIN{printf "%.3f", b / (1024 * 1024)}'
}

estimate_hadamard_memory_mib() {
    local rows_a="$1"
    local cols_a="$2"
    local rows_b="$3"
    local cols_b="$4"
    local depth="$5"
    local density_a="$6"
    local density_b="$7"

    local entries_a entries_b entries_out nnz_a nnz_b out_density nnz_out
    entries_a=$(storage_size_of "$rows_a" "$cols_a" "$depth")
    entries_b=$(storage_size_of "$rows_b" "$cols_b" "$depth")
    entries_out=$(storage_size_of "$rows_a" "$cols_b" "$depth")
    nnz_a=$(estimate_nnz "$entries_a" "$density_a")
    nnz_b=$(estimate_nnz "$entries_b" "$density_b")
    out_density=$(awk -v a="$density_a" -v b="$density_b" 'BEGIN{printf "%.12f", a * b}')
    nnz_out=$(estimate_nnz "$entries_out" "$out_density")

    local input_sparse_bytes working_sparse_bytes dense_working_bytes
    input_sparse_bytes=$(awk -v a="$nnz_a" -v b="$nnz_b" 'BEGIN{printf "%.0f", (a + b) * 24.0}')
    working_sparse_bytes=$(awk -v a="$nnz_a" -v b="$nnz_b" -v o="$nnz_out" 'BEGIN{printf "%.0f", (a + b + o) * 24.0}')
    dense_working_bytes=$(awk -v a="$entries_a" -v b="$entries_b" -v o="$entries_out" 'BEGIN{printf "%.0f", (a + b + o) * 8.0}')

    echo "$(bytes_to_mib "$input_sparse_bytes"),$(bytes_to_mib "$working_sparse_bytes"),$(bytes_to_mib "$dense_working_bytes")"
}

scale_aspect_dim() {
    local base="$1"
    if [ "$base" -le 500 ]; then
        echo $((base * 2))
    elif [ "$base" -le 1500 ]; then
        echo $((base * 3 / 2))
    elif [ "$base" -le 3000 ]; then
        echo $((base * 4 / 3))
    else
        echo $((base * 5 / 4))
    fi
}

resolve_shape_dims() {
    local base="$1"
    local shape_class="$2"

    local scaled
    scaled=$(scale_aspect_dim "$base")

    case "$shape_class" in
        square)
            echo "$base,$base,$base,$base"
            ;;
        tall)
            # Tall output slices: (rows_a >> cols_b)
            echo "$scaled,$base,$base,$base"
            ;;
        wide)
            # Wide output slices: (cols_b >> rows_a)
            echo "$base,$base,$base,$scaled"
            ;;
        *)
            return 1
            ;;
    esac
}

run_hadamard_case() {
    local mode="$1"
    local threads="$2"
    local rows_a="$3"
    local cols_a="$4"
    local rows_b="$5"
    local cols_b="$6"
    local depth="$7"
    local log_file="$8"

    local args=(
        --operation hadamard
        --rows-a "$rows_a" --cols-a "$cols_a"
        --rows-b "$rows_b" --cols-b "$cols_b"
        --depth "$depth"
        --density-a "$DENSITY" --density-b "$DENSITY"
        --seed 54321
        --warmup "$WARMUP"
        --timing-only
    )

    local bin="$CPU_BIN"
    case "$mode" in
        serial)
            export OMP_NUM_THREADS=1
            args+=(--mode serial)
            ;;
        omp)
            export OMP_NUM_THREADS="$threads"
            args+=(--mode omp --omp-chunk "$CHUNK_AUTO")
            ;;
        cuda)
            export OMP_NUM_THREADS=1
            bin="$CUDA_RUNNER"
            args+=(--mode cuda)
            ;;
        hybrid)
            export OMP_NUM_THREADS="$threads"
            bin="$CUDA_RUNNER"
            args+=(--mode hybrid --gpu-ratio "$GPU_RATIO" --omp-chunk "$CHUNK_AUTO")
            ;;
        *)
            return 1
            ;;
    esac

    if ! srun "$bin" "${args[@]}" > "$log_file" 2>&1; then
        return 1
    fi

    local time_sec iterations
    time_sec=$(extract_time "$log_file")
    iterations=$(extract_metric_default iterations "$log_file" "0")
    if [ -z "$time_sec" ]; then
        return 1
    fi

    local flops dense_slices sparse_slices dense_sec sparse_sec
    local work_units_total work_units_dense work_units_sparse
    local max_thread_work min_thread_work thread_work_imbalance
    local max_thread_time min_thread_time thread_time_imbalance
    local gpu_assigned_slices cpu_assigned_slices gpu_assigned_work cpu_assigned_work gpu_assigned_ratio partition_load_imbalance
    local storage_slices_ell storage_slices_coo
    local storage_dense_mib storage_coo_mib storage_ell_mib storage_active_mib
    local storage_comp_dense_to_coo storage_comp_dense_to_ell storage_comp_dense_to_active storage_ell_util

    flops=$(extract_metric_default flops "$log_file" "0")
    dense_slices=$(extract_metric_default dense_slices "$log_file" "0")
    sparse_slices=$(extract_metric_default sparse_slices "$log_file" "0")
    dense_sec=$(extract_metric_default dense_sec "$log_file" "0")
    sparse_sec=$(extract_metric_default sparse_sec "$log_file" "0")
    work_units_total=$(extract_metric_default work_units_total "$log_file" "0")
    work_units_dense=$(extract_metric_default work_units_dense "$log_file" "0")
    work_units_sparse=$(extract_metric_default work_units_sparse "$log_file" "0")
    max_thread_work=$(extract_metric_default max_thread_work "$log_file" "0")
    min_thread_work=$(extract_metric_default min_thread_work "$log_file" "0")
    thread_work_imbalance=$(extract_metric_default thread_work_imbalance "$log_file" "0")
    max_thread_time=$(extract_metric_default max_thread_time "$log_file" "0")
    min_thread_time=$(extract_metric_default min_thread_time "$log_file" "0")
    thread_time_imbalance=$(extract_metric_default thread_time_imbalance "$log_file" "0")
    gpu_assigned_slices=$(extract_metric_default gpu_assigned_slices "$log_file" "0")
    cpu_assigned_slices=$(extract_metric_default cpu_assigned_slices "$log_file" "0")
    gpu_assigned_work=$(extract_metric_default gpu_assigned_work "$log_file" "0")
    cpu_assigned_work=$(extract_metric_default cpu_assigned_work "$log_file" "0")
    gpu_assigned_ratio=$(extract_metric_default gpu_assigned_ratio "$log_file" "0")
    partition_load_imbalance=$(extract_metric_default partition_load_imbalance "$log_file" "0")

    storage_slices_ell=$(extract_storage_metric_default slices_ell "$log_file" "0")
    storage_slices_coo=$(extract_storage_metric_default slices_coo "$log_file" "0")
    storage_dense_mib=$(extract_storage_metric_default dense_mem_mib "$log_file" "0")
    storage_coo_mib=$(extract_storage_metric_default coo_mem_mib "$log_file" "0")
    storage_ell_mib=$(extract_storage_metric_default ell_mem_mib "$log_file" "0")
    storage_active_mib=$(extract_storage_metric_default active_mem_mib "$log_file" "0")
    storage_comp_dense_to_coo=$(extract_storage_metric_default compression_dense_to_coo "$log_file" "0")
    storage_comp_dense_to_ell=$(extract_storage_metric_default compression_dense_to_ell "$log_file" "0")
    storage_comp_dense_to_active=$(extract_storage_metric_default compression_dense_to_active "$log_file" "0")
    storage_ell_util=$(extract_storage_metric_default ell_slot_utilization "$log_file" "0")

    echo "$time_sec,$iterations,$flops,$dense_slices,$sparse_slices,$dense_sec,$sparse_sec,$work_units_total,$work_units_dense,$work_units_sparse,$max_thread_work,$min_thread_work,$thread_work_imbalance,$max_thread_time,$min_thread_time,$thread_time_imbalance,$gpu_assigned_slices,$cpu_assigned_slices,$gpu_assigned_work,$cpu_assigned_work,$gpu_assigned_ratio,$partition_load_imbalance,$storage_slices_ell,$storage_slices_coo,$storage_dense_mib,$storage_coo_mib,$storage_ell_mib,$storage_active_mib,$storage_comp_dense_to_coo,$storage_comp_dense_to_ell,$storage_comp_dense_to_active,$storage_ell_util"
}

validate_cuda_runner() {
    local smoke_log="$OUTPUT_DIR/cuda_smoke_test.log"
    local smoke_args=(
        --mode cuda
        --operation hadamard
        --rows-a 256 --cols-a 256
        --rows-b 256 --cols-b 256
        --depth 8
        --density-a 1.0 --density-b 1.0
        --seed 12345
        --warmup 0
        --timing-only
    )

    if ! srun "$CUDA_RUNNER" "${smoke_args[@]}" > "$smoke_log" 2>&1; then
        echo "[ERROR] CUDA smoke test failed; refusing CPU fallback." >&2
        tail -n 40 "$smoke_log" >&2 || true
        exit 1
    fi

    if ! grep -q "operation=hadamard time_sec=" "$smoke_log"; then
        echo "[ERROR] CUDA smoke test did not produce timing output." >&2
        tail -n 40 "$smoke_log" >&2 || true
        exit 1
    fi

    local dense_slices
    dense_slices=$(extract_metric dense_slices "$smoke_log")
    if [ -z "$dense_slices" ] || ! awk -v v="$dense_slices" 'BEGIN{exit (v > 0) ? 0 : 1}'; then
        echo "[ERROR] CUDA smoke test did not execute dense CUDA slices." >&2
        tail -n 40 "$smoke_log" >&2 || true
        exit 1
    fi

    echo "[INFO] CUDA smoke test passed with dense_slices=$dense_slices"
}

run_scalability_suite() {
    echo "====== Running Scalability Matrix ======"
    echo "Sparsity=$SPARSITY (density=$DENSITY), GPU_RATIO=$GPU_RATIO"
    echo "Tensor classes: square, tall, wide"
    echo "Base range: 100 to 5000 with depth 3 to 100"
    echo "OpenMP scheduling: dynamic (runtime), omp-chunk=${CHUNK_AUTO} (0 means auto-resolved in app)"
    validate_cuda_runner

    local shape_classes=(square tall wide)

    for idx in "${!ROWS_SERIES[@]}"; do
        local base depth
        base=${ROWS_SERIES[$idx]}
        depth=${DEPTH_SERIES[$idx]}
        for shape_class in "${shape_classes[@]}"; do
            local rows_a cols_a rows_b cols_b
            IFS=',' read -r rows_a cols_a rows_b cols_b <<< "$(resolve_shape_dims "$base" "$shape_class")"

            local output_shape input_shape_a input_shape_b
            output_shape="${rows_a}x${cols_b}x${depth}"
            input_shape_a="${rows_a}x${cols_a}x${depth}"
            input_shape_b="${rows_b}x${cols_b}x${depth}"

            local storage_input storage_output
            storage_input=$(storage_size_pair_of "$rows_a" "$cols_a" "$rows_b" "$cols_b" "$depth")
            storage_output=$(storage_size_of "$rows_a" "$cols_b" "$depth")

            local mem_payload input_mem_mib working_mem_mib dense_working_mem_mib
            mem_payload=$(estimate_hadamard_memory_mib "$rows_a" "$cols_a" "$rows_b" "$cols_b" "$depth" "$DENSITY" "$DENSITY")
            IFS=',' read -r input_mem_mib working_mem_mib dense_working_mem_mib <<< "$mem_payload"

            echo "[SIZE] class=${shape_class}, output=${output_shape}, A=${input_shape_a}, B=${input_shape_b}, storage_input=${storage_input}, sparsity=${SPARSITY}, input_mem=${input_mem_mib}MiB, working_mem=${working_mem_mib}MiB, dense_equiv=${dense_working_mem_mib}MiB"

            local serial_log="$OUTPUT_DIR/${shape_class}_serial_${output_shape}.log"
            local serial_payload
            if ! serial_payload=$(run_hadamard_case serial 1 "$rows_a" "$cols_a" "$rows_b" "$cols_b" "$depth" "$serial_log"); then
                echo "[ERROR] serial run failed for class=${shape_class} output=${output_shape}" >&2
                tail -n 20 "$serial_log" >&2 || true
                exit 1
            fi

            local serial_time serial_iterations serial_flops serial_dense_slices serial_sparse_slices
            local serial_dense_sec serial_sparse_sec serial_work_total serial_work_dense serial_work_sparse
            local serial_max_thread_work serial_min_thread_work serial_thread_work_imbalance
            local serial_max_thread_time serial_min_thread_time serial_thread_time_imbalance
            local serial_gpu_assigned_slices serial_cpu_assigned_slices serial_gpu_assigned_work serial_cpu_assigned_work serial_gpu_assigned_ratio serial_partition_load_imbalance
            local serial_storage_slices_ell serial_storage_slices_coo serial_storage_dense_mib serial_storage_coo_mib serial_storage_ell_mib serial_storage_active_mib serial_storage_comp_dense_to_coo serial_storage_comp_dense_to_ell serial_storage_comp_dense_to_active serial_storage_ell_util
            IFS=',' read -r serial_time serial_iterations serial_flops serial_dense_slices serial_sparse_slices serial_dense_sec serial_sparse_sec serial_work_total serial_work_dense serial_work_sparse serial_max_thread_work serial_min_thread_work serial_thread_work_imbalance serial_max_thread_time serial_min_thread_time serial_thread_time_imbalance serial_gpu_assigned_slices serial_cpu_assigned_slices serial_gpu_assigned_work serial_cpu_assigned_work serial_gpu_assigned_ratio serial_partition_load_imbalance serial_storage_slices_ell serial_storage_slices_coo serial_storage_dense_mib serial_storage_coo_mib serial_storage_ell_mib serial_storage_active_mib serial_storage_comp_dense_to_coo serial_storage_comp_dense_to_ell serial_storage_comp_dense_to_active serial_storage_ell_util <<< "$serial_payload"

            local serial_tpi serial_gflops
            serial_tpi=$(calc_ratio "$serial_time" "$serial_iterations")
            serial_gflops=$(calc_gflops "$serial_flops" "$serial_time")
            echo "serial,1,0,${shape_class},${output_shape},${input_shape_a},${input_shape_b},${depth},${storage_input},${storage_output},${SPARSITY},none,0,${input_mem_mib},${working_mem_mib},${dense_working_mem_mib},${serial_time},${serial_iterations},${serial_tpi},1.0,${serial_gflops},${serial_dense_slices},${serial_sparse_slices},${serial_dense_sec},${serial_sparse_sec},${serial_work_total},${serial_work_dense},${serial_work_sparse},${serial_max_thread_work},${serial_min_thread_work},${serial_thread_work_imbalance},${serial_max_thread_time},${serial_min_thread_time},${serial_thread_time_imbalance},${serial_gpu_assigned_slices},${serial_cpu_assigned_slices},${serial_gpu_assigned_work},${serial_cpu_assigned_work},${serial_gpu_assigned_ratio},${serial_partition_load_imbalance},${serial_storage_slices_ell},${serial_storage_slices_coo},${serial_storage_dense_mib},${serial_storage_coo_mib},${serial_storage_ell_mib},${serial_storage_active_mib},${serial_storage_comp_dense_to_coo},${serial_storage_comp_dense_to_ell},${serial_storage_comp_dense_to_active},${serial_storage_ell_util}" >> "$RESULT_CSV"

            local cuda_log="$OUTPUT_DIR/${shape_class}_cuda_${output_shape}.log"
            local cuda_payload
            if ! cuda_payload=$(run_hadamard_case cuda 1 "$rows_a" "$cols_a" "$rows_b" "$cols_b" "$depth" "$cuda_log"); then
                echo "[ERROR] cuda run failed for class=${shape_class} output=${output_shape}" >&2
                tail -n 20 "$cuda_log" >&2 || true
                exit 1
            fi

            local cuda_time cuda_iterations cuda_flops cuda_dense_slices cuda_sparse_slices
            local cuda_dense_sec cuda_sparse_sec cuda_work_total cuda_work_dense cuda_work_sparse
            local cuda_max_thread_work cuda_min_thread_work cuda_thread_work_imbalance
            local cuda_max_thread_time cuda_min_thread_time cuda_thread_time_imbalance
            local cuda_gpu_assigned_slices cuda_cpu_assigned_slices cuda_gpu_assigned_work cuda_cpu_assigned_work cuda_gpu_assigned_ratio cuda_partition_load_imbalance
            local cuda_storage_slices_ell cuda_storage_slices_coo cuda_storage_dense_mib cuda_storage_coo_mib cuda_storage_ell_mib cuda_storage_active_mib cuda_storage_comp_dense_to_coo cuda_storage_comp_dense_to_ell cuda_storage_comp_dense_to_active cuda_storage_ell_util
            IFS=',' read -r cuda_time cuda_iterations cuda_flops cuda_dense_slices cuda_sparse_slices cuda_dense_sec cuda_sparse_sec cuda_work_total cuda_work_dense cuda_work_sparse cuda_max_thread_work cuda_min_thread_work cuda_thread_work_imbalance cuda_max_thread_time cuda_min_thread_time cuda_thread_time_imbalance cuda_gpu_assigned_slices cuda_cpu_assigned_slices cuda_gpu_assigned_work cuda_cpu_assigned_work cuda_gpu_assigned_ratio cuda_partition_load_imbalance cuda_storage_slices_ell cuda_storage_slices_coo cuda_storage_dense_mib cuda_storage_coo_mib cuda_storage_ell_mib cuda_storage_active_mib cuda_storage_comp_dense_to_coo cuda_storage_comp_dense_to_ell cuda_storage_comp_dense_to_active cuda_storage_ell_util <<< "$cuda_payload"

            local cuda_tpi cuda_speedup cuda_gflops
            cuda_tpi=$(calc_ratio "$cuda_time" "$cuda_iterations")
            cuda_speedup=$(calc_ratio "$serial_time" "$cuda_time")
            cuda_gflops=$(calc_gflops "$cuda_flops" "$cuda_time")
            echo "cuda,1,0,${shape_class},${output_shape},${input_shape_a},${input_shape_b},${depth},${storage_input},${storage_output},${SPARSITY},none,0,${input_mem_mib},${working_mem_mib},${dense_working_mem_mib},${cuda_time},${cuda_iterations},${cuda_tpi},${cuda_speedup},${cuda_gflops},${cuda_dense_slices},${cuda_sparse_slices},${cuda_dense_sec},${cuda_sparse_sec},${cuda_work_total},${cuda_work_dense},${cuda_work_sparse},${cuda_max_thread_work},${cuda_min_thread_work},${cuda_thread_work_imbalance},${cuda_max_thread_time},${cuda_min_thread_time},${cuda_thread_time_imbalance},${cuda_gpu_assigned_slices},${cuda_cpu_assigned_slices},${cuda_gpu_assigned_work},${cuda_cpu_assigned_work},${cuda_gpu_assigned_ratio},${cuda_partition_load_imbalance},${cuda_storage_slices_ell},${cuda_storage_slices_coo},${cuda_storage_dense_mib},${cuda_storage_coo_mib},${cuda_storage_ell_mib},${cuda_storage_active_mib},${cuda_storage_comp_dense_to_coo},${cuda_storage_comp_dense_to_ell},${cuda_storage_comp_dense_to_active},${cuda_storage_ell_util}" >> "$RESULT_CSV"

            for mode in omp hybrid; do
                local mode_min_time=""
                local mode_max_time=""
                local best_threads=""
                local worst_threads=""
                local best_thread_work_imbalance=""
                local worst_thread_work_imbalance=""
                local best_thread_time_imbalance=""
                local worst_thread_time_imbalance=""

                for threads in "${THREAD_LIST[@]}"; do
                    local log_file="$OUTPUT_DIR/${shape_class}_${mode}_${output_shape}_t${threads}.log"
                    local payload
                    if ! payload=$(run_hadamard_case "$mode" "$threads" "$rows_a" "$cols_a" "$rows_b" "$cols_b" "$depth" "$log_file"); then
                        echo "[ERROR] ${mode} run failed for class=${shape_class} output=${output_shape}, threads=${threads}" >&2
                        tail -n 20 "$log_file" >&2 || true
                        exit 1
                    fi

                    local time_sec iterations flops dense_slices sparse_slices dense_sec sparse_sec
                    local work_units_total work_units_dense work_units_sparse
                    local max_thread_work min_thread_work thread_work_imbalance
                    local max_thread_time min_thread_time thread_time_imbalance
                    local gpu_assigned_slices cpu_assigned_slices gpu_assigned_work cpu_assigned_work gpu_assigned_ratio partition_load_imbalance
                    local storage_slices_ell storage_slices_coo storage_dense_mib storage_coo_mib storage_ell_mib storage_active_mib storage_comp_dense_to_coo storage_comp_dense_to_ell storage_comp_dense_to_active storage_ell_util
                    IFS=',' read -r time_sec iterations flops dense_slices sparse_slices dense_sec sparse_sec work_units_total work_units_dense work_units_sparse max_thread_work min_thread_work thread_work_imbalance max_thread_time min_thread_time thread_time_imbalance gpu_assigned_slices cpu_assigned_slices gpu_assigned_work cpu_assigned_work gpu_assigned_ratio partition_load_imbalance storage_slices_ell storage_slices_coo storage_dense_mib storage_coo_mib storage_ell_mib storage_active_mib storage_comp_dense_to_coo storage_comp_dense_to_ell storage_comp_dense_to_active storage_ell_util <<< "$payload"

                    local tpi speedup gflops thread_oversub
                    tpi=$(calc_ratio "$time_sec" "$iterations")
                    speedup=$(calc_ratio "$serial_time" "$time_sec")
                    gflops=$(calc_gflops "$flops" "$time_sec")
                    thread_oversub=0
                    if [ "$threads" -gt "$ALLOC_THREADS" ]; then
                        thread_oversub=1
                    fi

                    echo "${mode},${threads},${thread_oversub},${shape_class},${output_shape},${input_shape_a},${input_shape_b},${depth},${storage_input},${storage_output},${SPARSITY},dynamic,${CHUNK_AUTO},${input_mem_mib},${working_mem_mib},${dense_working_mem_mib},${time_sec},${iterations},${tpi},${speedup},${gflops},${dense_slices},${sparse_slices},${dense_sec},${sparse_sec},${work_units_total},${work_units_dense},${work_units_sparse},${max_thread_work},${min_thread_work},${thread_work_imbalance},${max_thread_time},${min_thread_time},${thread_time_imbalance},${gpu_assigned_slices},${cpu_assigned_slices},${gpu_assigned_work},${cpu_assigned_work},${gpu_assigned_ratio},${partition_load_imbalance},${storage_slices_ell},${storage_slices_coo},${storage_dense_mib},${storage_coo_mib},${storage_ell_mib},${storage_active_mib},${storage_comp_dense_to_coo},${storage_comp_dense_to_ell},${storage_comp_dense_to_active},${storage_ell_util}" >> "$RESULT_CSV"

                    if [ -z "$mode_min_time" ] || is_less "$time_sec" "$mode_min_time"; then
                        mode_min_time="$time_sec"
                        best_threads="$threads"
                        best_thread_work_imbalance="$thread_work_imbalance"
                        best_thread_time_imbalance="$thread_time_imbalance"
                    fi
                    if [ -z "$mode_max_time" ] || is_greater "$time_sec" "$mode_max_time"; then
                        mode_max_time="$time_sec"
                        worst_threads="$threads"
                        worst_thread_work_imbalance="$thread_work_imbalance"
                        worst_thread_time_imbalance="$thread_time_imbalance"
                    fi
                done

                local sweep_imbalance
                sweep_imbalance=$(calc_load_imbalance "$mode_max_time" "$mode_min_time")
                echo "${mode},${shape_class},${output_shape},${input_shape_a},${input_shape_b},${depth},${storage_input},${SPARSITY},${working_mem_mib},${mode_min_time},${mode_max_time},${sweep_imbalance},${best_threads},${worst_threads},${best_thread_work_imbalance},${worst_thread_work_imbalance},${best_thread_time_imbalance},${worst_thread_time_imbalance}" >> "$LOAD_CSV"
            done
        done
    done

    echo ""
    echo "====== Scalability Matrix Complete ======"
    echo "Main results: $RESULT_CSV"
    echo "Load imbalance: $LOAD_CSV"
    head -n 30 "$RESULT_CSV"
    head -n 30 "$LOAD_CSV"
}

run_scalability_suite
