module Hilbert

using CUDA
using Random

export MomentumSpinorBasis, initialize_states!, generate_proposal!

"""
波数(運動量)表示・スピン f=1 のボゾン多体系ヒルベルト空間
"""
mutable struct MomentumSpinorBasis
    k_max::Int          # カットオフ波数
    n_modes::Int        # モード総数 (2 * k_max + 1)
    n_particles::Int    # 全粒子数 (N)
    n_walkers::Int      # 並列ウォーカー数 (バッチサイズ)
    threads::Int
    
    # 状態配列: サイズ [n_modes, スピン成分数(3), n_walkers]
    # モードインデックス: 1 => -k_max, ..., (k_max+1) => 0, ..., n_modes => +k_max
    # スピン成分インデックス: 1 => m=-1, 2 => m=0, 3 => m=+1
    states::CuArray{Int32, 3} 

    function MomentumSpinorBasis(k_max::Int, n_particles::Int, threads::Int, n_walkers::Int)
        n_modes = 2 * k_max + 1
        states = CUDA.zeros(Int32, n_modes, 3, n_walkers)
        new(k_max, n_modes, n_particles, n_walkers, threads, states)
    end
end

"""
指定された全粒子数 N と 総磁化 M_z を満たす初期状態を生成する
- basis: MomentumSpinorBasis のインスタンス
- target_Mz: 目標とする総磁化 M_z の値 (例: 0, -N, +N など)
"""
function initialize_states!(basis::MomentumSpinorBasis, target_Mz::Int)
    N = basis.n_particles
    L = basis.n_modes
    zero = div(L - 1, 2) + 1
    
    # 物理的要件のチェック
    if abs(target_Mz) > N * 1
        error("総磁化 M_z の絶対値は全磁化 N * 1 を超えることはできません。")
    end
    
    # CPU上で初期状態のベース (L × 3行列) を作成
    base_state = zeros(Int32, L, 3)
    base_state[zero, 2] = N - abs(target_Mz)   # sz = 0
    if target_Mz >= 0
        base_state[zero, 3] = target_Mz         # sz = +1
    else
        base_state[zero, 1] = -target_Mz        # sz = -1 (負のMzで負の占有数を書かない)
    end

    # 作成したベース状態を、全ウォーカー(バッチ)分コピーして3次元配列にする
    # repmatのような操作を行い、CPU上で [L, 3, n_walkers] の配列を作る
    initial_states_cpu = repeat(base_state, 1, 1, basis.n_walkers)
    
    # 最後にGPU (CUDA) へ一括転送して構造体の状態を上書き
    copyto!(basis.states, CuArray(initial_states_cpu))
    
    return nothing
end

"""
全ウォーカーに対して、ランダムな2体散乱を提案する関数
- states: 現在の状態配列 [n_modes, 3, n_walkers]
- proposed_states: 提案状態を書き込むための配列（同じサイズ）
- k_max: カットオフ波数
"""
function generate_proposal!(states::CuArray{Int32, 3}, proposed_states::CuArray{Int32, 3}, h_factor::CuArray{Float32, 1}, k_max::Int, n_particles::Int, threads::Int)
    n_walkers = size(states, 3)
    
    # 各ウォーカーに対して4つの乱数を用意する
    # [1]: 粒子1の選択用, [2]: 粒子2の選択用, [3]: 運動量qの選択用, [4]: スピン交換の分岐用
    rand_vals = CUDA.rand(Float32, 4, n_walkers)
    tmp_states = CUDA.zeros(Int32, 2 * k_max + 1, 3, n_walkers)
    
    blocks = ceil(Int, n_walkers / threads)
    @cuda threads=threads blocks=blocks _proposal_kernel!(
        states, proposed_states, tmp_states, h_factor, rand_vals, k_max, n_particles
    )
    ## _proposal_kernel!(
    ##     states, proposed_states, h_factor, rand_vals, k_max
    ## )
 
    return nothing
end

"""
各ウォーカーごとに独立して1つのランダムな2体散乱を提案するカーネル
"""
function _proposal_kernel!(states, proposed_states, tmp_states, h_factor, rand_vals, k_max, n_particles)
    # ウォーカーのインデックスを指定
    w = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_modes = 2 * k_max + 1
    
    if w <= size(states, 3)

        # 1. 状態のコピー
        for m in 1:n_modes, s in 1:3
            proposed_states[m, s, w] = states[m, s, w]
            tmp_states[m, s, w] = states[m, s, w]
        end

        # 2. ターゲットの波数・スピンを選択            
        m1, s1 = k_max + 1, 2
        # CUDA.rand (cuRAND) は区間 (0,1] で 1.0f0 を含むため、target が
        # n_particles を超えると選択ループがフォールスルーして状態が壊れる。クランプで防ぐ。
        target = min(trunc(Int32, rand_vals[1, w] * n_particles) + 1, n_particles)
        count = Int32(0)
        for m in 1:n_modes, s in 1:3
            occ = tmp_states[m, s, w]
            if occ > 0
                prev_count = count
                count += occ
                if prev_count < target <= count; m1 = m; s1 = s; break; end
            end
        end
        tmp_states[m1, s1, w] -= 1
        
        m2, s2 = k_max + 1, 2
        target = min(trunc(Int32, rand_vals[2, w] * (n_particles - 1)) + 1, n_particles - 1)
        count = Int32(0)
        for m in 1:n_modes, s in 1:3
            occ = tmp_states[m, s, w]
            if occ > 0
                prev_count = count
                count += occ
                if prev_count < target <= count; m2 = m; s2 = s; break; end
            end
        end
        tmp_states[m2, s2, w] -= 1
        
        # 3. スピンの交換
        s1_new, s2_new = s1, s2
        rand_val = rand_vals[3, w]
        if (s1 == 2 && s2 == 2) && rand_val < 0.5f0
            s1_new, s2_new = 1, 3
        elseif (s1 == 1 && s2 == 3) && rand_val < 0.5f0
            s1_new, s2_new = 2, 2
        elseif (s1 == 3 && s2 == 1) && rand_val < 0.5f0
            s1_new, s2_new = 2, 2
        else
            s1_new, s2_new = s1, s2
        end

        # 4. 散乱後の波数を計算
        q = floor(Int, rand_vals[4, w] * n_modes) - k_max
        k1 = m1 - k_max - 1
        k2 = m2 - k_max - 1
        k1_new = k1 + q
        k2_new = k2 - q
        m1_new = k1_new + k_max + 1
        m2_new = k2_new + k_max + 1

        if m1_new < 1 || m2_new < 1 || m1_new > n_modes || m2_new > n_modes
            h_factor[w] = 1
            return
        end
        
        # 5. 移動先の粒子を増やす
        proposed_states[m1, s1, w] -= 1
        proposed_states[m2, s2, w] -= 1
        proposed_states[m1_new, s1_new, w] += 1
        proposed_states[m2_new, s2_new, w] += 1

        # 6. Hastings因子の計算
        factor = 1f0
        if s1 == s2 && m1 == m2
            factor /= states[m1, s1, w] * (states[m2, s2, w] - 1)
        else
            factor /= states[m1, s1, w] * states[m2, s2, w] * 2
        end

        if s1_new == s2_new && m1_new == m2_new
            factor *= proposed_states[m1_new, s1_new, w] * (proposed_states[m2_new, s2_new, w] - 1)
        else
            factor *= proposed_states[m1_new, s1_new, w] * proposed_states[m2_new, s2_new, w] * 2
        end

        # ---- 追加: q の縮退度補正 ----
        # 順方向: 行き先が同スピン・異運動量なら同じ終状態を与える q が2通り
        if s1_new == s2_new && m1_new != m2_new
            factor /= 2f0
        end
        # 逆方向: 元の2軌道が同スピン・異運動量なら逆過程の q が2通り
        if s1 == s2 && m1 != m2
            factor *= 2f0
        end
 
        h_factor[w] = factor
    end

    return nothing
end

end # module Hilbert