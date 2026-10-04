#!/usr/bin/env bash
#SBATCH --job-name=tensor_debug
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --nodes=1-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --output=debug_%j.out
#SBATCH --error=debug_%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND} (exit=${rc})" >&2; exit ${rc}' ERR

echo "[INFO] job_id=${SLURM_JOB_ID:-na} host=$(hostname) submit_dir=${SLURM_SUBMIT_DIR:-na} pwd=$PWD"
echo "[INFO] Running all test modes: correctness, debug, determinism"

ROOT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
if [ ! -f "$ROOT_DIR/Makefile" ]; then
    ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
fi
cd "$ROOT_DIR"
echo "[INFO] root_dir=$ROOT_DIR"

CPU_BIN="$ROOT_DIR/tensor_app_cpu"
CUDA_BIN="$ROOT_DIR/tensor_app_cuda"
COMPARE="$ROOT_DIR/scripts/compare_tensors.py"
OUTPUT_DIR="$ROOT_DIR/artifacts/debug"
mkdir -p "$OUTPUT_DIR"
echo "[INFO] output_dir=$OUTPUT_DIR"

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

PASS=0
FAIL=0
CHUNK_AUTO=0
DEFAULT_GPU_RATIO=0.5

pass() {
    echo "[PASS] $1"
    PASS=$((PASS + 1))
}

fail() {
    echo "[FAIL] $1"
    FAIL=$((FAIL + 1))
}

run_case() {
    local label="$1"
    shift
    if "$@"; then
        pass "$label"
    else
        fail "$label"
    fi
}

run_and_capture() {
    local log_file="$1"
    shift
    : > "$log_file"
    if "$@" > "$log_file" 2>&1; then
        return 0
    fi
    return 1
}

timed_srun() {
    local log_file="$1"
    shift
    local start_t end_t
    start_t=$(date +%s%N)
    if ! run_and_capture "$log_file" srun "$@"; then
        return 1
    fi
    end_t=$(date +%s%N)
    echo "scale=6; ($end_t - $start_t) / 1000000000" | bc
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

MODE_BIN=""
MODE_ARGS=()

configure_mode() {
    local mode="$1"
    local omp_chunk="${2:-$CHUNK_AUTO}"
    local gpu_ratio="${3:-$DEFAULT_GPU_RATIO}"

    MODE_ARGS=()
    case "$mode" in
        serial)
            MODE_BIN="$CPU_BIN"
            MODE_ARGS=(--mode serial)
            ;;
        omp)
            MODE_BIN="$CPU_BIN"
            MODE_ARGS=(--mode omp --omp-chunk "$omp_chunk")
            ;;
        cuda)
            MODE_BIN="$CUDA_RUNNER"
            MODE_ARGS=(--mode cuda)
            ;;
        hybrid)
            MODE_BIN="$CUDA_RUNNER"
            MODE_ARGS=(--mode hybrid --gpu-ratio "$gpu_ratio" --omp-chunk "$omp_chunk")
            ;;
        *)
            return 1
            ;;
    esac
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

# Numeric tolerance for cross-mode floating-point comparisons in large debug runs.
COMPARE_TOL_DEBUG=1e-6
COMPARE_TOL_SMALL=1e-8

CSV_FILE="$OUTPUT_DIR/correctness_results.csv"
echo "density_combo,operation,mode,storage_format,tensor_shape,storage_size,sparsity_a,sparsity_b,input_mem_mib,working_set_mem_mib,dense_equiv_working_set_mem_mib,time_sec,executions_total,input_slices_ell,input_slices_coo,input_dense_mem_mib,input_coo_mem_mib,input_ell_mem_mib,input_active_mem_mib,input_compression_dense_to_coo,input_compression_dense_to_ell,input_compression_dense_to_active,input_ell_slot_utilization" > "$CSV_FILE"

test_combination() {
    local dens_a=$1
    local dens_b=$2
    local combo_name=$3
    local seed=$4
    local rows=$5
    local cols=$6
    local depth=$7
    local warmup=$8
    local modes=(serial omp cuda hybrid)
    local mode_names=("serial" "openmp" "cuda" "hybrid")

    local shape storage_size sparsity_a sparsity_b mem_payload input_mem_mib working_mem_mib dense_working_mem_mib
    shape="${rows}x${cols}x${depth}"
    storage_size=$(storage_size_of "$rows" "$cols" "$depth")
    sparsity_a=$(awk -v d="$dens_a" 'BEGIN{printf "%.6f", 1.0 - d}')
    sparsity_b=$(awk -v d="$dens_b" 'BEGIN{printf "%.6f", 1.0 - d}')
    mem_payload=$(estimate_hadamard_memory_mib "$rows" "$cols" "$depth" "$dens_a" "$dens_b")
    IFS=',' read -r input_mem_mib working_mem_mib dense_working_mem_mib <<< "$mem_payload"

    echo "Testing combination: ${combo_name} (density_a=${dens_a}, density_b=${dens_b})"
    echo "[MEM] ${combo_name}: shape=${shape}, storage_size=${storage_size}, sparsity_a=${sparsity_a}, sparsity_b=${sparsity_b}, input_mem=${input_mem_mib}MiB, working_mem=${working_mem_mib}MiB, dense_equiv=${dense_working_mem_mib}MiB"

    local serial_out="$OUTPUT_DIR/${combo_name}_hadamard_serial.csv"
    for idx in "${!modes[@]}"; do
        local m=${modes[$idx]}
        local mname=${mode_names[$idx]}
        local out_file="$OUTPUT_DIR/${combo_name}_hadamard_${mname}.csv"

        local args=(--operation hadamard --rows-a $rows --cols-a $cols --rows-b $rows --cols-b $cols --depth $depth --density-a "$dens_a" --density-b "$dens_b" --seed "$seed" --warmup "$warmup")
        configure_mode "$m" "$CHUNK_AUTO" "$DEFAULT_GPU_RATIO"
        args+=("${MODE_ARGS[@]}")

        local log_file="$OUTPUT_DIR/${combo_name}_hadamard_${mname}.log"
        rm -f "$out_file" "$log_file"
        local elapsed_sec
        if ! elapsed_sec=$(timed_srun "$log_file" "$MODE_BIN" "${args[@]}" --out "$out_file"); then
            fail "${combo_name}: hadamard ${mname} execution failed"
            tail -n 20 "$log_file" || true
            continue
        fi

        if [ ! -s "$out_file" ]; then
            fail "${combo_name}: hadamard ${mname} produced empty output"
            tail -n 20 "$log_file" || true
            continue
        fi

        if [ "$m" = "serial" ]; then
            serial_out="$out_file"
            run_case "${combo_name}: hadamard ${mname} executes" [ -f "$serial_out" ]
        else
            if [ ! -s "$serial_out" ]; then
                fail "${combo_name}: hadamard serial baseline missing"
                continue
            fi
            run_case "${combo_name}: hadamard ${mname} matches serial" python3 "$COMPARE" "$serial_out" "$out_file" "$COMPARE_TOL_DEBUG"
        fi

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

        echo "${combo_name},hadamard,${mname},auto,${shape},${storage_size},${sparsity_a},${sparsity_b},${input_mem_mib},${working_mem_mib},${dense_working_mem_mib},${elapsed_sec},1,${storage_slices_ell},${storage_slices_coo},${storage_dense_mib},${storage_coo_mib},${storage_ell_mib},${storage_active_mib},${storage_comp_dense_to_coo},${storage_comp_dense_to_ell},${storage_comp_dense_to_active},${storage_ell_util}" >> "$CSV_FILE"
    done

    if [ -f "$serial_out" ]; then
        for op in qr svd; do
            local op_label="QR"
            if [ "$op" = "svd" ]; then
                op_label="SVD"
            fi

            local format_labels=(coo ell)
            local format_thresholds=(1.10 0.0)

            for format_idx in "${!format_labels[@]}"; do
                local storage_format=${format_labels[$format_idx]}
                local preprocess_threshold=${format_thresholds[$format_idx]}

                for idx in "${!modes[@]}"; do
                    local m=${modes[$idx]}
                    local mname=${mode_names[$idx]}

                    local args=(--operation "$op" --input-a "$serial_out" --rows-a $rows --cols-a $cols --rows-b $rows --cols-b $cols --depth $depth --preprocess-threshold "$preprocess_threshold" --warmup "$warmup" --timing-only)
                    configure_mode "$m" "$CHUNK_AUTO" "$DEFAULT_GPU_RATIO"
                    args+=("${MODE_ARGS[@]}")

                    local log_file="$OUTPUT_DIR/${combo_name}_${op}_${mname}_${storage_format}.log"
                    rm -f "$log_file"
                    local elapsed_sec
                    if ! elapsed_sec=$(timed_srun "$log_file" "$MODE_BIN" "${args[@]}"); then
                        fail "${combo_name}: ${op_label} ${mname} (${storage_format}) execution failed"
                        tail -n 20 "$log_file" || true
                        continue
                    fi

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

                    run_case "${combo_name}: ${op_label} ${mname} (${storage_format}) executes" grep -q "operation=${op} time_sec=" "$log_file"
                    echo "${combo_name},${op},${mname},${storage_format},${shape},${storage_size},${sparsity_a},${sparsity_b},${input_mem_mib},${working_mem_mib},${dense_working_mem_mib},${elapsed_sec},1,${storage_slices_ell},${storage_slices_coo},${storage_dense_mib},${storage_coo_mib},${storage_ell_mib},${storage_active_mib},${storage_comp_dense_to_coo},${storage_comp_dense_to_ell},${storage_comp_dense_to_active},${storage_ell_util}" >> "$CSV_FILE"
                done
            done
        done
    fi
}

run_correctness_mode() {
    local SMALL_ROWS=50
    local SMALL_COLS=50
    local SMALL_DEPTH=3
    local WARMUP=1

    echo "====== Running correctness tests on 50x50x3 tensors ======"

    run_hadamard_case() {
        local tag="$1"
        local density_a="$2"
        local density_b="$3"
        local seed="$4"
        local modes=(serial omp cuda hybrid)

        local serial_out="$OUTPUT_DIR/${tag}_serial.csv"
        local common_args=(--operation hadamard --rows-a $SMALL_ROWS --cols-a $SMALL_COLS --rows-b $SMALL_ROWS --cols-b $SMALL_COLS --depth $SMALL_DEPTH --density-a "$density_a" --density-b "$density_b" --seed "$seed" --warmup $WARMUP)

        print_case_memory "$tag" "$SMALL_ROWS" "$SMALL_COLS" "$SMALL_DEPTH" "$density_a" "$density_b"
        echo "Running Hadamard case: ${tag} (density_a=${density_a}, density_b=${density_b})"

        for mode in "${modes[@]}"; do
            local out_file="$OUTPUT_DIR/${tag}_${mode}.csv"
            local log_file="$OUTPUT_DIR/${tag}_hadamard_${mode}.log"

            configure_mode "$mode" "$CHUNK_AUTO" 0.6
            if ! timed_srun "$log_file" "$MODE_BIN" "${MODE_ARGS[@]}" "${common_args[@]}" --out "$out_file" >/dev/null; then
                fail "${tag}: hadamard ${mode} execution failed"
                tail -n 20 "$log_file" || true
                continue
            fi

            if [ "$mode" = "serial" ]; then
                serial_out="$out_file"
                run_case "${tag}: hadamard serial executes" [ -s "$serial_out" ]
            else
                run_case "${tag}: serial vs ${mode} hadamard match" python3 "$COMPARE" "$serial_out" "$out_file" "$COMPARE_TOL_SMALL"
            fi
        done

        local decomp_args=(--input-a "$serial_out" --rows-a $SMALL_ROWS --cols-a $SMALL_COLS --rows-b $SMALL_ROWS --cols-b $SMALL_COLS --depth $SMALL_DEPTH --warmup 1 --timing-only)
        for op in qr svd; do
            local op_label="QR"
            if [ "$op" = "svd" ]; then
                op_label="SVD"
            fi

            local format_labels=(coo ell)
            local format_thresholds=(1.10 0.0)

            for format_idx in "${!format_labels[@]}"; do
                local storage_format=${format_labels[$format_idx]}
                local preprocess_threshold=${format_thresholds[$format_idx]}

                for mode in "${modes[@]}"; do
                    local log_file="$OUTPUT_DIR/${tag}_${op}_${mode}_${storage_format}.log"
                    configure_mode "$mode" "$CHUNK_AUTO" 0.6
                    if ! timed_srun "$log_file" "$MODE_BIN" "${MODE_ARGS[@]}" --operation "$op" --preprocess-threshold "$preprocess_threshold" "${decomp_args[@]}" >/dev/null; then
                        fail "${tag}: ${op_label} ${mode} (${storage_format}) execution failed"
                        tail -n 20 "$log_file" || true
                        continue
                    fi
                    run_case "${tag}: ${op_label} ${mode} (${storage_format}) executes" grep -q "operation=${op} time_sec=" "$log_file"
                done
            done
        done
    }

    run_hadamard_case "sparse_sparse" 0.05 0.05 101
    run_hadamard_case "dense_dense" 1.0 1.0 202
    run_hadamard_case "sparse_dense" 0.05 1.0 303

    echo "------------------------------------------------------------"
    echo "Correctness summary: PASS=$PASS FAIL=$FAIL"
    echo "------------------------------------------------------------"
}

run_debug_mode() {
    local rows=1000
    local cols=1000
    local depth=10
    local warmup=2

    echo "====== Running debug tests on 1000x1000x10 tensors ======"

    test_combination 0.005 0.005 "sparse_sparse" 101 "$rows" "$cols" "$depth" "$warmup"
    test_combination 1.0 1.0 "dense_dense" 202 "$rows" "$cols" "$depth" "$warmup"
    test_combination 0.005 1.0 "sparse_dense" 303 "$rows" "$cols" "$depth" "$warmup"
    test_combination 1.0 0.005 "dense_sparse" 404 "$rows" "$cols" "$depth" "$warmup"

    echo ""
    echo "=========================================="
    echo "Debug summary: PASS=$PASS FAIL=$FAIL"
    echo "Results written to: $CSV_FILE"
    echo "=========================================="
}

run_determinism_mode() {
    local WARMUP=1
    echo "Running determinism check on sparse_sparse case..."
    print_case_memory "determinism_sparse_sparse" 50 50 3 0.05 0.05

    local DET_ARGS=(--operation hadamard --rows-a 50 --cols-a 50 --rows-b 50 --cols-b 50 --depth 3 --density-a 0.05 --density-b 0.05 --seed 404 --warmup $WARMUP)
    srun "$CPU_BIN" --mode serial "${DET_ARGS[@]}" --out "$OUTPUT_DIR/determinism_run1.csv"
    srun "$CPU_BIN" --mode serial "${DET_ARGS[@]}" --out "$OUTPUT_DIR/determinism_run2.csv"
    run_case "determinism: serial run1 vs run2 match" python3 "$COMPARE" "$OUTPUT_DIR/determinism_run1.csv" "$OUTPUT_DIR/determinism_run2.csv" 1e-12
    echo "Determinism summary: PASS=$PASS FAIL=$FAIL"
}

echo "====== Running all test modes ======"

validate_cuda_runner

TEST_MODES=(correctness debug determinism)
TOTAL_FAIL=0

for TEST_MODE in "${TEST_MODES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Starting test mode: $TEST_MODE"
    echo "=========================================="

    PASS=0
    FAIL=0

    case "$TEST_MODE" in
        correctness)
            export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-36}
            run_correctness_mode
            ;;
        debug)
            export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-36}
            run_debug_mode
            ;;
        determinism)
            export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-36}
            run_determinism_mode
            ;;
    esac

    echo "Mode: $TEST_MODE - PASS=$PASS FAIL=$FAIL"
    TOTAL_FAIL=$((TOTAL_FAIL + FAIL))
done

echo ""
echo "====== All test modes complete ======"
echo "Total failures: $TOTAL_FAIL"

if [ "$TOTAL_FAIL" -gt 0 ]; then
    exit 1
fi
