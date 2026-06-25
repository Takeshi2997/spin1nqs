module Physics

using CUDA
using Lux
using ..Model
# using ..Hilbert # 必要に応じて

export SystemParams, PhysicsBuffer, compute_local_energy!

"""
系の物理パラメータをまとめる構造体
"""
struct SystemParams
    k_max::Int
    n_modes::Int
    hbar2_over_2m::Float32
    
    # 相互作用パラメータ
    c0::Float32
    c1::Float32
end

"""
局所エネルギー計算用のGPUメモリを使い回すためのバッファ
"""
struct PhysicsBuffer
    proposed_states::CuArray{Int32, 4}      # [n_modes, 3, MAX_TRANSITIONS, n_walkers]
    matrix_elements::CuArray{Float32, 2}    # [MAX_TRANSITIONS, n_walkers]
    
    function PhysicsBuffer(k_max::Int, max_transitions::Int, n_walkers::Int)
        n_modes = 2 * k_max + 1
        
        # 最初に1回だけ巨大なGPUメモリを確保する
        proposed_states = CUDA.zeros(Int32, n_modes, 3, max_transitions, n_walkers)
        matrix_elements = CUDA.zeros(Float32, max_transitions, n_walkers)
        
        new(proposed_states, matrix_elements)
    end
end

"""
運動エネルギー（対角項）を計算する関数
"""
function compute_kinetic(states::CuArray{Int32, 3}, params::SystemParams)
    n_modes, _, n_walkers = size(states)
    
    # 波数ベクトル k = [-k_max, ..., k_max] を作成し、二乗する
    k_vec = CuArray(Float32.(-params.k_max:params.k_max))
    k2_vec = k_vec .^ 2 # サイズ: [n_modes]
    
    # statesの形状 [n_modes, 3, n_walkers] に対して、各モードの粒子数とk^2を掛けて足し合わせる
    # sum(..., dims=(1,2)) でモードとスピン成分について和をとる
    E_kin = sum(states .* k2_vec, dims=(1, 2)) .* params.hbar2_over_2m
    
    # 戻り値は [1, 1, n_walkers] のテンソルを [n_walkers] のベクトルにして返す
    return dropdims(E_kin, dims=(1, 2)) 
end

"""
局所エネルギー全体を評価するメイン関数
"""
function compute_local_energy!(
    states::CuArray{Int32, 3}, 
    log_psi_current::CuArray{ComplexF32, 1}, 
    proposed_states::CuArray{Int32, 4},
    matrix_elements::CuArray{Float32, 2},
    params::SystemParams, 
    threads::Int,
    model, ps, st
)
    n_walkers = size(states, 3)
    
    # 1. 運動エネルギーの計算（一瞬で終わります）
    E_loc = compute_kinetic(states, params) 
    
    # 2. 相互作用エネルギーの計算（遷移テンソルの構築とNN評価）
    
    # (A) 遷移先状態 x' と、その行列要素 V_xx' を一括生成する関数
    # proposed_states: [n_modes, 3, max_transitions, n_walkers]
    # matrix_elements: [max_transitions, n_walkers]
    blocks = ceil(Int, n_walkers / threads)
    @cuda threads=threads blocks=blocks _scattering_kernel!(
        states, 
        proposed_states, 
        matrix_elements, 
        params.k_max,
        params.c0,
        params.c1
    )
    ## _scattering_kernel!(
    ##     states, 
    ##     proposed_states, 
    ##     matrix_elements, 
    ##     params.k_max,
    ##     params.c0,
    ##     params.c1
    ## )
    
    # (B) 提案状態をネットワークに通すため、バッチ次元を平坦化
    # [n_modes, 3, max_transitions * n_walkers] に変形
    flat_proposed = reshape(proposed_states, params.n_modes, 3, :)

    # (C) ネットワークで一括評価
    flat_log_psi_prop = eval_complex_network(model, flat_proposed, ps, st)
    
    # (D) 元の形 [max_transitions, n_walkers] に戻す
    log_psi_prop = reshape(flat_log_psi_prop, size(matrix_elements, 1), n_walkers)
    
    # (E) 波動関数の比 Ψ(x')/Ψ(x) = exp(log_psi_prop - log_psi_current) を計算し、行列要素と掛ける
    # Ensure log_psi_current is a 1 x n_walkers row for broadcasting
    psi_ratio = exp.(log_psi_prop .- reshape(log_psi_current, 1, :))
    
    # 相互作用エネルギー E_V = sum_{x'} V_xx' * (Ψ(x')/Ψ(x)) を足し合わせる
    E_V = sum(matrix_elements .* psi_ratio, dims=1)

    # 3. 合計して返す
    E_loc += dropdims(E_V, dims=1)
    
    return E_loc
end

"""
全ウォーカーのすべての2体散乱プロセスを列挙し、遷移先状態と行列要素を書き込むカーネル
"""
function _scattering_kernel!(
    states,             # [n_modes, 3, n_walkers] 現在の状態
    proposed_states,    # [n_modes, 3, MAX_TRANSITIONS, n_walkers] 遷移先を書き込むバッファ
    matrix_elements,    # [MAX_TRANSITIONS, n_walkers] 行列要素 V_xx' を書き込むバッファ
    k_max::Int, c0::Float32, c1::Float32
)
    n_modes = 2 * k_max + 1
    w = (blockIdx().x - 1) * blockDim().x + threadIdx().x # 自分が担当するウォーカーID

    if w <= size(states, 3)
        transition_idx = 1
        
        # 定数部分
        v0 = c0 / (2.0f0 * π)
        v1 = c1 / (2.0f0 * π)

        # 散乱する2粒子のモードを選択
        for m1 in 1:n_modes, s1 in 1:3
            n1 = states[m1, s1, w]
            if n1 == 0; continue; end
            
            for m2 in 1:n_modes, s2 in 1:3
                n2 = states[m2, s2, w]
                if n2 == 0; continue; end
                
                # 同一モード・同一スピンから2つ選ぶ場合は、2個以上いる必要がある
                if (m1 == m2 && s1 == s2) && n2 < 2; continue; end
                
                # 現在の運動量
                k1 = m1 - k_max - 1
                k2 = m2 - k_max - 1
                
                # 遷移する前(消滅演算子)の因子
                factor_annihilate = Float32(n1) * Float32(m1 == m2 && s1 == s2 ? n2 - 1 : n2)
 
                # 運動量移動 q のループ
                for q in -k_max:k_max
                    # 散乱後の運動量
                    k1_new = k1 + q
                    k2_new = k2 - q

                    m1_new = k1_new + k_max + 1
                    m2_new = k2_new + k_max + 1

                    if m1_new >= 1 && m1_new <= n_modes && m2_new >= 1 && m2_new <= n_modes
                                               
                        # 状態をコピーして更新
                        _update_proposed_states!(proposed_states, states, n_modes, m1, s1, m2, s2, m1_new, s1, m2_new, s2, transition_idx, w)
                        
                        # 行列要素の計算
                        bose_factor = _calculate_bose_factor(proposed_states, factor_annihilate, m1_new, s1, m2_new, s2, transition_idx, w)
                        matrix_elements[transition_idx, w] = (v0 + v1 * s1 * s2) / 2 * bose_factor

                        transition_idx += 1

                        # (0, 0) <--> (1, -1) の遷移
                        if s1 == 2 && s2 == 2
                            s1_new, s2_new = 1, 3
                            
                            # 状態をコピーして更新
                            _update_proposed_states!(proposed_states, states, n_modes, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, transition_idx, w)

                            # 行列要素の計算
                            bose_factor = _calculate_bose_factor(proposed_states, factor_annihilate, m1_new, s1_new, m2_new, s2_new, transition_idx, w)
                            matrix_elements[transition_idx, w] = v1 * bose_factor

                            transition_idx += 1

                        # (1, -1) <--> (0, 0), (-1, 1) <--> (0, 0) の遷移
                        elseif (s1 == 1 && s2 == 3) || (s1 == 3 && s2 == 1)
                            s1_new, s2_new = 2, 2
                            
                            # 状態をコピーして更新
                            _update_proposed_states!(proposed_states, states, n_modes, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, transition_idx, w)
                            
                            # 行列要素の計算
                            bose_factor = _calculate_bose_factor(proposed_states, factor_annihilate, m1_new, s1_new, m2_new, s2_new, transition_idx, w)
                            matrix_elements[transition_idx, w] = 2 * v1 * bose_factor

                            transition_idx += 1
                        
                        # スピンの交換
                        elseif abs(s1 - s2) == 1
                            s1_new, s2_new = s2, s1
                            
                            # 状態をコピーして更新
                            _update_proposed_states!(proposed_states, states, n_modes, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, transition_idx, w)
                            
                            # スピン交換用のボース統計因子を計算
                            bose_factor = _calculate_bose_factor(proposed_states, factor_annihilate, m1_new, s1_new, m2_new, s2_new, transition_idx, w)
                            matrix_elements[transition_idx, w] = v1 * bose_factor

                            transition_idx += 1
                        end
                    end
                end
            end
        end

        # 使わなかった残りのスロットは行列要素を 0 にして無効化する
        while transition_idx <= size(matrix_elements, 1)
            matrix_elements[transition_idx, w] = 0.0f0
            transition_idx += 1
        end
    end
    return nothing
end

"""
Updates the proposed states based on the current states and transition parameters.
"""
function _update_proposed_states!(proposed_states, states, n_modes, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, id, w)
    # 状態をコピーして更新S
    for s in 1:3
        for m in 1:n_modes
            proposed_states[m, s, id, w] = states[m, s, w]
        end
    end
    proposed_states[m1, s1, id, w] -= 1
    proposed_states[m2, s2, id, w] -= 1
    proposed_states[m1_new, s1_new, id, w] += 1
    proposed_states[m2_new, s2_new, id, w] += 1
end

"""
Calculates the Bose statistics factor for a given transition.
"""
function _calculate_bose_factor(proposed_states, factor_annihilate, m1_new, s1_new, m2_new, s2_new, id, w)
    n1_new = proposed_states[m1_new, s1_new, id, w]
    n2_new = proposed_states[m2_new, s2_new, id, w]
    factor_create = Float32(n1_new) * Float32(m1_new == m2_new && s1_new == s2_new ? n2_new - 1 : n2_new)

    return sqrt(factor_annihilate * factor_create)
end

end # module