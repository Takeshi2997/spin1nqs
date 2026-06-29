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
    base_state[zero, 2] = N - target_Mz
    base_state[zero, 3] = target_Mz

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
    
    blocks = ceil(Int, n_walkers / threads)
    
    @cuda threads=threads blocks=blocks _proposal_kernel!(
        states, proposed_states, h_factor, rand_vals, k_max, n_particles
    )
    ## _proposal_kernel!(
    ##     states, proposed_states, h_factor, rand_vals, k_max, n_particles
    ## )
 
    return nothing
end

"""
各ウォーカーごとに独立して1つのランダムな2体散乱を提案するカーネル
"""
function _proposal_kernel!(states, proposed_states, h_factor, rand_vals, k_max, n_particles)
    # ウォーカーのインデックスを指定
    w = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n_modes = 2 * k_max + 1
    
    if w <= size(states, 3)

        # 1. 状態のコピー（初期化）
        for m in 1:n_modes, s in 1:3
            proposed_states[m, s, w] = states[m, s, w]
        end

        # 2. 粒子をランダムに選ぶ
        target1 = ceil(Int32, rand_vals[1, w] * n_particles)
        target2 = ceil(Int32, rand_vals[2, w] * n_particles)
        # 同じ粒子を選んでしまった場合は遷移しない
        if target1 == target2; return; end 
        
        # 配列を走査して粒子を特定
        m1, s1, m2, s2 = 0, 0, 0, 0
        count = Int32(0)
        for m in 1:n_modes, s in 1:3
            occ = states[m, s, w]
            if occ > 0
                prev_count = count
                count += occ
                if prev_count < target1 <= count; m1 = m; s1 = s; end
                if prev_count < target2 <= count; m2 = m; s2 = s; end
            end
        end

        # 3. 運動量移動 q の決定
        q = floor(Int, rand_vals[3, w] * n_modes) - k_max

        k1 = m1 - k_max - 1
        k2 = m2 - k_max - 1
        
        k1_new = k1 + q
        k2_new = k2 - q
        if abs(k1_new) > k_max || abs(k2_new) > k_max
            return
        end
        m1_new = k1_new + k_max + 1
        m2_new = k2_new + k_max + 1

	    # 4. スピン交換プロセスの処理 (確率的に分岐)
        # 50%の確率で m_f=1 (s=1) と m_f=-1 (s=3) に遷移させる
        s1_new, s2_new = s1, s2
        if s1 == 2 && s2 == 2
            if rand_vals[4, w] < 0.5f0
                s1_new, s2_new = 1, 3
                _update_proposed_states(proposed_states, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, w)
                h_factor[w] = _calculate_h_factor(states, proposed_states, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, w)
            end
        elseif (s1 == 1 && s2 == 3) || (s1 == 3 && s2 == 1)
            if rand_vals[4, w] < 0.5f0
                s1_new, s2_new = 2, 2
                _update_proposed_states(proposed_states, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, w)
                h_factor[w] = _calculate_h_factor(states, proposed_states, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, w)
            end
        end

        if s1_new == s1 && s2_new == s2
            _update_proposed_states(proposed_states, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, w)
            h_factor[w] = _calculate_h_factor(states, proposed_states, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, w)
        end

    end

    return nothing
end

function _update_proposed_states(proposed_states, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, w)
    # 選んだ粒子を減らす
    proposed_states[m1, s1, w] -= 1
    proposed_states[m2, s2, w] -= 1
    # 移動先の粒子を増やす
    proposed_states[m1_new, s1_new, w] += 1
    proposed_states[m2_new, s2_new, w] += 1
end

function _calculate_h_factor(states, proporsed_states, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, w)
    factor = 1
    n1 = states[m1, s1, w]
    n2 = states[m2, s2, w]
    if (m1 == m2) && (s1 == s2)
        factor *= n1 * (n2 - 1)
    else
        factor *= n1 * n2
    end

    n1_new = proporsed_states[m1_new, s1_new, w]
    n2_new = proporsed_states[m2_new, s2_new, w]
    if (m1_new == m2_new) && (s1_new == s2_new)
        factor /= n1_new * (n2_new - 1)
    else
        factor /= n1_new * n2_new
    end

    return factor
end
 

end # module Hilbert