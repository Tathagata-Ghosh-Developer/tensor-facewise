#!/usr/bin/env bash
#SBATCH --job-name=tensor_profile
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --nodes=1-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --output=profile_%j.out
#SBATCH --error=profile_%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND} (exit=${rc})" >&2; exit ${rc}' ERR

echo "[INFO] job_id=${SLURM_JOB_ID:-na} host=$(hostname) submit_dir=${SLURM_SUBMIT_DIR:-na} pwd=$PWD"
echo "[INFO] Running all profiling modes: tools, bottleneck"

ROOT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
if [ ! -f "$ROOT_DIR/Makefile" ]; then
    ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
fi
cd "$ROOT_DIR"
echo "[INFO] root_dir=$ROOT_DIR"

CPU_BIN="$ROOT_DIR/tensor_app_cpu"
CUDA_BIN="$ROOT_DIR/tensor_app_cuda"
ARTIFACT_DIR="$ROOT_DIR/artifacts/profile"
mkdir -p "$ARTIFACT_DIR"
echo "[INFO] artifact_dir=$ARTIFACT_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-36}
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
    module load tau >/dev/null 2>&1 || module load TAU >/dev/null 2>&1 || true
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

prepend_path() {
    local path_to_add="$1"
    if [ ! -d "$path_to_add" ]; then
        return
    fi
    case ":${PATH:-}:" in
        *":${path_to_add}:"*)
            ;;
        *)
            export PATH="${path_to_add}:${PATH:-}"
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

storage_metric_payload() {
    local log_file="$1"
    local storage_slices_ell storage_slices_coo
    local storage_dense_mib storage_coo_mib storage_ell_mib storage_active_mib
    local storage_comp_dense_to_coo storage_comp_dense_to_ell storage_comp_dense_to_active storage_ell_util

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

    echo "${storage_slices_ell},${storage_slices_coo},${storage_dense_mib},${storage_coo_mib},${storage_ell_mib},${storage_active_mib},${storage_comp_dense_to_coo},${storage_comp_dense_to_ell},${storage_comp_dense_to_active},${storage_ell_util}"
}

validate_cuda_runner() {
    local smoke_log="$ARTIFACT_DIR/cuda_smoke_test.log"
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

    if ! srun "$PROFILE_BIN" "${smoke_args[@]}" > "$smoke_log" 2>&1; then
        echo "[ERROR] CUDA smoke test failed for profiling." >&2
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

resolve_tau_exec() {
    if [ -n "${TAU_EXEC:-}" ] && [ -x "${TAU_EXEC}" ]; then
        echo "$TAU_EXEC"
        return 0
    fi

    if command -v tau_exec >/dev/null 2>&1; then
        command -v tau_exec
        return 0
    fi

    local candidates=(
        "${TAUROOTDIR:-}/x86_64/bin/tau_exec"
        "${TAUROOTDIR:-}/bin/tau_exec"
        "${TAU_ROOT:-}/x86_64/bin/tau_exec"
        "${TAU_ROOT:-}/bin/tau_exec"
        "/opt/tau/x86_64/bin/tau_exec"
        "/opt/tau/bin/tau_exec"
        "/usr/local/tau/x86_64/bin/tau_exec"
        "$HOME/tau/bin/tau_exec"
        "$HOME/.local/tau/bin/tau_exec"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        [ -n "$candidate" ] || continue
        if [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

configure_tau_environment() {
    local tau_roots=(
        "${TAUROOTDIR:-}"
        "${TAU_ROOT:-}"
        "/opt/tau"
        "/usr/local/tau"
    )

    local root
    for root in "${tau_roots[@]}"; do
        [ -n "$root" ] || continue
        [ -d "$root" ] || continue

        export TAU_ROOT="$root"
        export TAUROOTDIR="$root"
        prepend_path "$root/x86_64/bin"
        prepend_path "$root/bin"

        if [ -z "${TAU_MAKEFILE:-}" ]; then
            if [ -f "$root/x86_64/lib/Makefile.tau-mpi-openmp" ]; then
                export TAU_MAKEFILE="$root/x86_64/lib/Makefile.tau-mpi-openmp"
            elif [ -f "$root/lib/Makefile.tau-mpi-openmp" ]; then
                export TAU_MAKEFILE="$root/lib/Makefile.tau-mpi-openmp"
            fi
        fi
        break
    done

    export TAU_PROFILE=${TAU_PROFILE:-1}
    export TAU_CALLPATH=${TAU_CALLPATH:-1}
    export TAU_CALLPATH_DEPTH=${TAU_CALLPATH_DEPTH:-10}
}

make -C "$ROOT_DIR" cpu
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[ERROR] nvcc not found in PATH; CUDA profiling is required for this script." >&2
    exit 1
fi
make -C "$ROOT_DIR" cuda

if [ ! -x "$CUDA_BIN" ]; then
    echo "[ERROR] CUDA binary missing or not executable: $CUDA_BIN" >&2
    exit 1
fi
discover_cuda_runtime_dirs
ensure_cuda_binary_runtime "$CUDA_BIN"
PROFILE_BIN="$CUDA_BIN"

configure_tau_environment

TAU_REQUIRED=${TAU_REQUIRED:-0}
TAU_EXEC=""
if TAU_EXEC=$(resolve_tau_exec 2>/dev/null); then
    echo "[INFO] TAU executable detected: $TAU_EXEC"
    echo "[INFO] TAU_ROOT=${TAU_ROOT:-unset}"
    echo "[INFO] TAU_MAKEFILE=${TAU_MAKEFILE:-unset}"
    echo "[INFO] TAU_CALLPATH_DEPTH=${TAU_CALLPATH_DEPTH}"
else
    echo "[WARN] TAU is unavailable in current environment (no module and no tau_exec in PATH)."
    echo "[WARN] TAU profiling will be skipped. To verify cluster availability, run: module spider tau"
    if [ "$TAU_REQUIRED" = "1" ]; then
        echo "[ERROR] TAU_REQUIRED=1 but TAU is unavailable." >&2
        exit 1
    fi
fi

THREAD_CANDIDATES=(1 2 4 8 16 32 36 64 72)
ALLOC_THREADS=${SLURM_CPUS_PER_TASK:-36}
THREAD_LIST=()
for candidate in "${THREAD_CANDIDATES[@]}"; do
    if [ "$candidate" -le "$ALLOC_THREADS" ]; then
        THREAD_LIST+=("$candidate")
    fi
done
if [ "${#THREAD_LIST[@]}" -eq 0 ]; then
    THREAD_LIST=(1)
fi

echo "[INFO] profiling thread set=${THREAD_LIST[*]} (allocated_threads=${ALLOC_THREADS})"

storage_size_of() {
    local rows="$1"
    local cols="$2"
    local depth="$3"
    awk -v r="$rows" -v c="$cols" -v d="$depth" 'BEGIN{printf "%.0f", r * c * d}'
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
    local rows="$1"
    local cols="$2"
    local depth="$3"
    local density_a="$4"
    local density_b="$5"

    local entries nnz_a nnz_b out_density nnz_out
    entries=$(storage_size_of "$rows" "$cols" "$depth")
    nnz_a=$(estimate_nnz "$entries" "$density_a")
    nnz_b=$(estimate_nnz "$entries" "$density_b")
    out_density=$(awk -v a="$density_a" -v b="$density_b" 'BEGIN{printf "%.12f", a * b}')
    nnz_out=$(estimate_nnz "$entries" "$out_density")

    local input_sparse_bytes working_sparse_bytes dense_working_bytes
    input_sparse_bytes=$(awk -v a="$nnz_a" -v b="$nnz_b" 'BEGIN{printf "%.0f", (a + b) * 24.0}')
    working_sparse_bytes=$(awk -v a="$nnz_a" -v b="$nnz_b" -v o="$nnz_out" 'BEGIN{printf "%.0f", (a + b + o) * 24.0}')
    dense_working_bytes=$(awk -v e="$entries" 'BEGIN{printf "%.0f", 3.0 * e * 8.0}')

    echo "$(bytes_to_mib "$input_sparse_bytes"),$(bytes_to_mib "$working_sparse_bytes"),$(bytes_to_mib "$dense_working_bytes")"
}

print_case_memory() {
    local label="$1"
    local rows="$2"
    local cols="$3"
    local depth="$4"
    local density_a="$5"
    local density_b="$6"

    local storage_size sparsity_a sparsity_b mem_payload input_mem working_mem dense_equiv
    storage_size=$(storage_size_of "$rows" "$cols" "$depth")
    sparsity_a=$(awk -v d="$density_a" 'BEGIN{printf "%.6f", 1.0 - d}')
    sparsity_b=$(awk -v d="$density_b" 'BEGIN{printf "%.6f", 1.0 - d}')
    mem_payload=$(estimate_hadamard_memory_mib "$rows" "$cols" "$depth" "$density_a" "$density_b")
    IFS=',' read -r input_mem working_mem dense_equiv <<< "$mem_payload"

    echo "[MEM] ${label}: shape=${rows}x${cols}x${depth}, storage_size=${storage_size}, sparsity_a=${sparsity_a}, sparsity_b=${sparsity_b}, input_mem=${input_mem}MiB, working_mem=${working_mem}MiB, dense_equiv=${dense_equiv}MiB"
}

run_timed_command() {
    local log_file="$1"
    shift
    local start_t end_t
    start_t=$(date +%s%N)
    if ! "$@" > "$log_file" 2>&1; then
        return 1
    fi
    end_t=$(date +%s%N)
    echo "scale=6; ($end_t - $start_t) / 1000000000" | bc
}

run_tools_profiling() {
    local tools_dir="$ARTIFACT_DIR/tools"
    mkdir -p "$tools_dir"
    echo "[INFO] tools_dir=$tools_dir"

    local common_args=(
        --operation hadamard
        --rows-a 256 --cols-a 256
        --rows-b 256 --cols-b 256
        --depth 16
        --density-a 1.0 --density-b 1.0
        --seed 12345
        --warmup 1
        --timing-only
    )

    print_case_memory "tools_cuda_dense" 256 256 16 1.0 1.0

    if command -v nsys >/dev/null 2>&1; then
        echo "[1/3] Nsight Systems timeline"
        if ! nsys profile -t cuda,osrt,nvtx,openmp -o "$tools_dir/nsys_cuda_timeline" --force-overwrite true "$PROFILE_BIN" --mode cuda "${common_args[@]}"; then
            echo "[WARN] nsys run failed; continuing with remaining tools."
        fi
    else
        echo "[WARN] nsys not found; skipping NVIDIA timeline profiling"
    fi

    if command -v ncu >/dev/null 2>&1; then
        echo "[2/3] Nsight Compute kernel metrics"
        if ! ncu --set full -o "$tools_dir/ncu_cuda_kernels" --force-overwrite --target-processes all "$PROFILE_BIN" --mode cuda "${common_args[@]}"; then
            echo "[WARN] ncu run failed; continuing with remaining tools."
        fi
    else
        echo "[WARN] ncu not found; skipping CUDA kernel metric profiling"
    fi

    if [ -n "$TAU_EXEC" ] && [ -x "$TAU_EXEC" ]; then
        echo "[3/3] TAU profiling"
        export TAU_METRICS=${TAU_METRICS:-TIME,PAPI_L1_DCM}

        export PROFILEDIR="$tools_dir/tau_omp"
        mkdir -p "$PROFILEDIR"
        "$TAU_EXEC" -T serial,openmp "$CPU_BIN" --mode omp "${common_args[@]}"

        export PROFILEDIR="$tools_dir/tau_hybrid"
        mkdir -p "$PROFILEDIR"
        "$TAU_EXEC" -T serial,openmp "$PROFILE_BIN" --mode hybrid --gpu-ratio 0.7 "${common_args[@]}"

        echo "[INFO] TAU profiles saved to $tools_dir/tau_omp and $tools_dir/tau_hybrid"
    else
        echo "[WARN] tau_exec not found; skipping TAU profiling"
    fi

    echo "[INFO] profiling artifacts written to $tools_dir"
}

run_bottleneck_profiling() {
    local bottleneck_dir="$ARTIFACT_DIR/bottleneck"
    mkdir -p "$bottleneck_dir"
    echo "[INFO] bottleneck_dir=$bottleneck_dir"

    local warmup=2
    local summary="$bottleneck_dir/profile_summary.txt"
    local csv="$bottleneck_dir/bottleneck_metrics.csv"
    > "$summary"
    echo "case,mode,threads_or_ratio,tensor_shape,storage_size,sparsity,input_mem_mib,working_set_mem_mib,dense_equiv_working_set_mem_mib,time_sec,speedup_vs_serial,input_slices_ell,input_slices_coo,input_dense_mem_mib,input_coo_mem_mib,input_ell_mem_mib,input_active_mem_mib,input_compression_dense_to_coo,input_compression_dense_to_ell,input_compression_dense_to_active,input_ell_slot_utilization" > "$csv"

    echo "====== Profiling Selected Cases for Bottleneck Analysis ======"
    echo "Profile Summary Report" > "$summary"
    echo "======================" >> "$summary"

    local c1_rows=2000 c1_depth=20 c1_density=0.005
    local c1_shape="${c1_rows}x${c1_rows}x${c1_depth}"
    local c1_storage c1_mem_payload c1_input_mem c1_working_mem c1_dense_mem
    c1_storage=$(storage_size_of "$c1_rows" "$c1_rows" "$c1_depth")
    c1_mem_payload=$(estimate_hadamard_memory_mib "$c1_rows" "$c1_rows" "$c1_depth" "$c1_density" "$c1_density")
    IFS=',' read -r c1_input_mem c1_working_mem c1_dense_mem <<< "$c1_mem_payload"

    echo "[Case 1] Sparse OMP scaling: ${c1_shape}"
    print_case_memory "case1_sparse_omp" "$c1_rows" "$c1_rows" "$c1_depth" "$c1_density" "$c1_density"

    for threads in "${THREAD_LIST[@]}"; do
        export OMP_NUM_THREADS=$threads
        local log="$bottleneck_dir/case1_sparse_omp_t${threads}.log"
        local elapsed
        if ! elapsed=$(run_timed_command "$log" srun "$CPU_BIN" --operation hadamard --mode omp --omp-chunk 0 --rows-a "$c1_rows" --cols-a "$c1_rows" --rows-b "$c1_rows" --cols-b "$c1_rows" --depth "$c1_depth" --density-a "$c1_density" --density-b "$c1_density" --seed 54321 --warmup "$warmup" --timing-only); then
            echo "[ERROR] Case 1 failed for threads=${threads}" >&2
            tail -n 40 "$log" >&2 || true
            exit 1
        fi
        echo "  Threads=${threads}: wall_time=${elapsed}s" | tee -a "$summary"
        local c1_storage_payload
        c1_storage_payload=$(storage_metric_payload "$log")
        echo "case1_sparse_omp,omp,${threads},${c1_shape},${c1_storage},$(awk -v d="$c1_density" 'BEGIN{printf "%.6f", 1.0-d}'),${c1_input_mem},${c1_working_mem},${c1_dense_mem},${elapsed},NA,${c1_storage_payload}" >> "$csv"
    done

    local c2_rows=4000 c2_depth=12 c2_density=1.0
    local c2_shape="${c2_rows}x${c2_rows}x${c2_depth}"
    local c2_storage c2_mem_payload c2_input_mem c2_working_mem c2_dense_mem
    c2_storage=$(storage_size_of "$c2_rows" "$c2_rows" "$c2_depth")
    c2_mem_payload=$(estimate_hadamard_memory_mib "$c2_rows" "$c2_rows" "$c2_depth" "$c2_density" "$c2_density")
    IFS=',' read -r c2_input_mem c2_working_mem c2_dense_mem <<< "$c2_mem_payload"

    echo "[Case 2] Dense serial vs CUDA: ${c2_shape}"
    print_case_memory "case2_dense_serial_cuda" "$c2_rows" "$c2_rows" "$c2_depth" "$c2_density" "$c2_density"

    export OMP_NUM_THREADS=1
    local log_serial="$bottleneck_dir/case2_dense_serial.log"
    local serial_time
    if ! serial_time=$(run_timed_command "$log_serial" srun "$CPU_BIN" --operation hadamard --mode serial --rows-a "$c2_rows" --cols-a "$c2_rows" --rows-b "$c2_rows" --cols-b "$c2_rows" --depth "$c2_depth" --density-a "$c2_density" --density-b "$c2_density" --seed 54321 --warmup "$warmup" --timing-only); then
        echo "[ERROR] Case 2 serial run failed" >&2
        tail -n 40 "$log_serial" >&2 || true
        exit 1
    fi

    local log_cuda="$bottleneck_dir/case2_dense_cuda.log"
    local cuda_time
    if ! cuda_time=$(run_timed_command "$log_cuda" srun "$PROFILE_BIN" --operation hadamard --mode cuda --rows-a "$c2_rows" --cols-a "$c2_rows" --rows-b "$c2_rows" --cols-b "$c2_rows" --depth "$c2_depth" --density-a "$c2_density" --density-b "$c2_density" --seed 54321 --warmup "$warmup" --timing-only); then
        echo "[ERROR] Case 2 CUDA run failed" >&2
        tail -n 40 "$log_cuda" >&2 || true
        exit 1
    fi

    local speedup2
    speedup2=$(awk -v s="$serial_time" -v c="$cuda_time" 'BEGIN{if (c == 0) {print "0"} else {printf "%.4f", s / c}}')
    local c2_serial_storage_payload c2_cuda_storage_payload
    c2_serial_storage_payload=$(storage_metric_payload "$log_serial")
    c2_cuda_storage_payload=$(storage_metric_payload "$log_cuda")
    echo "case2_dense_compare,serial,1,${c2_shape},${c2_storage},0.000000,${c2_input_mem},${c2_working_mem},${c2_dense_mem},${serial_time},1.0000,${c2_serial_storage_payload}" >> "$csv"
    echo "case2_dense_compare,cuda,1,${c2_shape},${c2_storage},0.000000,${c2_input_mem},${c2_working_mem},${c2_dense_mem},${cuda_time},${speedup2},${c2_cuda_storage_payload}" >> "$csv"

    local c3_rows=8000 c3_depth=24 c3_density=0.001
    local c3_shape="${c3_rows}x${c3_rows}x${c3_depth}"
    local c3_storage c3_mem_payload c3_input_mem c3_working_mem c3_dense_mem
    c3_storage=$(storage_size_of "$c3_rows" "$c3_rows" "$c3_depth")
    c3_mem_payload=$(estimate_hadamard_memory_mib "$c3_rows" "$c3_rows" "$c3_depth" "$c3_density" "$c3_density")
    IFS=',' read -r c3_input_mem c3_working_mem c3_dense_mem <<< "$c3_mem_payload"

    echo "[Case 3] Larger sparse OMP scaling: ${c3_shape}"
    print_case_memory "case3_large_sparse_omp" "$c3_rows" "$c3_rows" "$c3_depth" "$c3_density" "$c3_density"

    for threads in "${THREAD_LIST[@]}"; do
        export OMP_NUM_THREADS=$threads
        local log="$bottleneck_dir/case3_sparse_omp_t${threads}.log"
        local elapsed
        if ! elapsed=$(run_timed_command "$log" srun "$CPU_BIN" --operation hadamard --mode omp --omp-chunk 0 --rows-a "$c3_rows" --cols-a "$c3_rows" --rows-b "$c3_rows" --cols-b "$c3_rows" --depth "$c3_depth" --density-a "$c3_density" --density-b "$c3_density" --seed 54321 --warmup "$warmup" --timing-only); then
            echo "[ERROR] Case 3 failed for threads=${threads}" >&2
            tail -n 40 "$log" >&2 || true
            exit 1
        fi
        echo "  Threads=${threads}: wall_time=${elapsed}s" | tee -a "$summary"
        local c3_storage_payload
        c3_storage_payload=$(storage_metric_payload "$log")
        echo "case3_large_sparse_omp,omp,${threads},${c3_shape},${c3_storage},$(awk -v d="$c3_density" 'BEGIN{printf "%.6f", 1.0-d}'),${c3_input_mem},${c3_working_mem},${c3_dense_mem},${elapsed},NA,${c3_storage_payload}" >> "$csv"
    done

    local c4_rows=6000 c4_depth=16 c4_density=0.01
    local c4_shape="${c4_rows}x${c4_rows}x${c4_depth}"
    local c4_storage c4_mem_payload c4_input_mem c4_working_mem c4_dense_mem
    c4_storage=$(storage_size_of "$c4_rows" "$c4_rows" "$c4_depth")
    c4_mem_payload=$(estimate_hadamard_memory_mib "$c4_rows" "$c4_rows" "$c4_depth" "$c4_density" "$c4_density")
    IFS=',' read -r c4_input_mem c4_working_mem c4_dense_mem <<< "$c4_mem_payload"

    echo "[Case 4] Hybrid split scan: ${c4_shape}"
    print_case_memory "case4_hybrid_ratio_scan" "$c4_rows" "$c4_rows" "$c4_depth" "$c4_density" "$c4_density"

    local last_thread_idx
    last_thread_idx=$(( ${#THREAD_LIST[@]} - 1 ))
    local hybrid_threads="${THREAD_LIST[$last_thread_idx]}"
    export OMP_NUM_THREADS="$hybrid_threads"

    for ratio in 0.3 0.5 0.7; do
        local log="$bottleneck_dir/case4_hybrid_ratio_${ratio}.log"
        local elapsed
        if ! elapsed=$(run_timed_command "$log" srun "$PROFILE_BIN" --operation hadamard --mode hybrid --gpu-ratio "$ratio" --omp-chunk 0 --rows-a "$c4_rows" --cols-a "$c4_rows" --rows-b "$c4_rows" --cols-b "$c4_rows" --depth "$c4_depth" --density-a "$c4_density" --density-b "$c4_density" --seed 54321 --warmup "$warmup" --timing-only); then
            echo "[ERROR] Case 4 failed for gpu_ratio=${ratio}" >&2
            tail -n 40 "$log" >&2 || true
            exit 1
        fi
        echo "  GPU_ratio=${ratio} (threads=${hybrid_threads}): wall_time=${elapsed}s" | tee -a "$summary"
        local c4_storage_payload
        c4_storage_payload=$(storage_metric_payload "$log")
        echo "case4_hybrid_ratio_scan,hybrid,${ratio},${c4_shape},${c4_storage},$(awk -v d="$c4_density" 'BEGIN{printf "%.6f", 1.0-d}'),${c4_input_mem},${c4_working_mem},${c4_dense_mem},${elapsed},NA,${c4_storage_payload}" >> "$csv"
    done

    echo "Summary report: $summary"
    echo "CSV report: $csv"
}

echo "====== Running all profiling modes ======"
validate_cuda_runner

profile_modes=(tools bottleneck)
for profile_mode in "${profile_modes[@]}"; do
    echo ""
    echo "=========================================="
    echo "Starting profiling mode: $profile_mode"
    echo "=========================================="

    case "$profile_mode" in
        tools)
            export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-36}
            run_tools_profiling
            ;;
        bottleneck)
            export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-36}
            run_bottleneck_profiling
            ;;
    esac

    echo "Mode: $profile_mode - Complete"
done

echo ""
echo "====== All profiling modes complete ======"
echo "Profiling complete."
