module Physics

using CUDA
using Lux
using ..Model
# using ..Hilbert # 必要に応じて

export SystemParams, PhysicsBuffer, compute_local_energy, compute_local_correlation, compute_local_density_matrix

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
function _compute_kinetic(states::CuArray{Int32, 3}, params::SystemParams)
    # 波数ベクトル k = [-k_max, ..., k_max] を作成し、二乗する
    k_vec = CuArray(Float32.(-params.k_max:params.k_max) .* 2 .* π ./ params.n_modes)
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
function compute_local_energy(
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
    E_loc = _compute_kinetic(states, params) 
    
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
        v0 = c0 / Float32(n_modes)
        v1 = c1 / Float32(n_modes)

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
                        matrix_elements[transition_idx, w] = (v0 + v1 * (s1 - 2) * (s2 - 2)) / 2 * bose_factor

                        transition_idx += 1

                        # (0, 0) <--> (1, -1) の遷移
                        if s1 == 2 && s2 == 2
                            s1_new, s2_new = 1, 3
                            
                            # 状態をコピーして更新
                            _update_proposed_states!(proposed_states, states, n_modes, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, transition_idx, w)

                            # 行列要素の計算
                            bose_factor = _calculate_bose_factor(proposed_states, factor_annihilate, m1_new, s1_new, m2_new, s2_new, transition_idx, w)
                            matrix_elements[transition_idx, w] = v1 / 2 * bose_factor

                            transition_idx += 1

                            s1_new, s2_new = 3, 1
                            
                            # 状態をコピーして更新
                            _update_proposed_states!(proposed_states, states, n_modes, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, transition_idx, w)

                            # 行列要素の計算
                            bose_factor = _calculate_bose_factor(proposed_states, factor_annihilate, m1_new, s1_new, m2_new, s2_new, transition_idx, w)
                            matrix_elements[transition_idx, w] = v1 / 2 * bose_factor

                            transition_idx += 1
                            
                        # (1, -1) <--> (0, 0), (-1, 1) <--> (0, 0) の遷移
                        elseif (s1 == 1 && s2 == 3) || (s1 == 3 && s2 == 1)
                            s1_new, s2_new = 2, 2
                            
                            # 状態をコピーして更新
                            _update_proposed_states!(proposed_states, states, n_modes, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, transition_idx, w)
                            
                            # 行列要素の計算
                            bose_factor = _calculate_bose_factor(proposed_states, factor_annihilate, m1_new, s1_new, m2_new, s2_new, transition_idx, w)
                            matrix_elements[transition_idx, w] = v1 / 2 * bose_factor

                            transition_idx += 1
                            
                        # スピンの交換
                        elseif abs(s1 - s2) == 1
                            s1_new, s2_new = s2, s1
                            
                            # 状態をコピーして更新
                            _update_proposed_states!(proposed_states, states, n_modes, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, transition_idx, w)
                            
                            # スピン交換用のボース統計因子を計算
                            bose_factor = _calculate_bose_factor(proposed_states, factor_annihilate, m1_new, s1_new, m2_new, s2_new, transition_idx, w)
                            matrix_elements[transition_idx, w] = v1 / 2 * bose_factor

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
    # 状態をコピーして更新
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

function compute_local_correlation(
    states::CuArray{Int32, 3}, 
    log_psi_current::CuArray{ComplexF32, 1}, 
    k_max::Int,
    threads::Int,
    model, ps, st
)
    n_walkers = size(states, 3)
    n_modes = 2 * k_max + 1
    n_q = 2 * k_max            # q = -k_max..-1, +1..+k_max (q=0 は対角項なので別扱い)
    max_t_per = n_modes^2      # 1つの (s, q) ブロックが必要とするスロット数の上限
    total_t = 3 * n_q * max_t_per

    # 通しインデックス total_t で proposed_states と matrix_elements を 1対1 対応させる
    proposed_states = CUDA.zeros(Int32, n_modes, 3, total_t, n_walkers)
    matrix_elements = CUDA.zeros(Float32, total_t, n_walkers)
 
    # (A) 遷移先状態 x' と行列要素を一括生成
    blocks = ceil(Int, n_walkers / threads)
    @cuda threads=threads blocks=blocks _correlation_kernel!(
        states, 
        proposed_states, 
        matrix_elements, 
        k_max,
        max_t_per,
    )

    # (B) 提案状態をネットワークに通すため、バッチ次元を平坦化
    flat_proposed = reshape(proposed_states, n_modes, 3, :)

    # (C) ネットワークで一括評価
    flat_log_psi_prop = eval_complex_network(model, flat_proposed, ps, st)
    
    # (D) 元の形 [total_t, n_walkers] に戻す
    log_psi_prop = reshape(flat_log_psi_prop, total_t, n_walkers)
    
    # (E) 各スロットの寄与 = 行列要素 × そのスロット自身の遷移先の ψ 比
    psi_ratio = exp.(log_psi_prop .- reshape(log_psi_current, 1, :))
    contrib = matrix_elements .* psi_ratio                       # [total_t, n_walkers]

    # (F) ブロック構造 [max_t_per, n_q, 3, n_walkers] に戻し、スロット方向に和をとる
    contrib = reshape(contrib, max_t_per, n_q, 3, n_walkers)
    rho2_qs = dropdims(sum(contrib, dims=1), dims=1)             # [n_q, 3, n_walkers]

    # (G) 旧来の [n_modes(=q平面), 3, n_walkers] レイアウトに詰め直す (q=0 平面はゼロのまま)
    rho2_q = CUDA.zeros(ComplexF32, n_modes, 3, n_walkers)
    rho2_q[1:k_max, :, :] .= rho2_qs[1:k_max, :, :]    # q = -k_max..-1
    rho2_q[k_max+2:n_modes, :, :] .= rho2_qs[k_max+1:n_q, :, :] # q = +1..+k_max
    
    return rho2_q
end


"""
全ウォーカーのすべての2体散乱プロセスを列挙し、運動量空間の相関を計算するカーネル
"""
function _correlation_kernel!(
    states,             # [n_modes, 3, n_walkers] 現在の状態
    proposed_states,    # [n_modes, 3, total_t, n_walkers] 遷移先を書き込むバッファ
    matrix_elements,    # [total_t, n_walkers] 行列要素を書き込むバッファ
    k_max::Int,
    max_t_per::Int      # (s, q) ブロックあたりのスロット数 (= n_modes^2)
)
    n_modes = 2 * k_max + 1
    n_q = 2 * k_max
    w = (blockIdx().x - 1) * blockDim().x + threadIdx().x # 自分が担当するウォーカーID

    if w <= size(states, 3)
        for s in 1:3
            for qi in 1:n_q
                # qi = 1..k_max -> q = -k_max..-1,  qi = k_max+1..2k_max -> q = +1..+k_max
                q = qi <= k_max ? qi - k_max - 1 : qi - k_max
                base = ((s - 1) * n_q + (qi - 1)) * max_t_per
                t = 1

                for m1 in 1:n_modes
                    n1 = states[m1, s, w]
                    if n1 == 0; continue; end
                    
                    for m2 in 1:n_modes
                        n2 = states[m2, s, w]
                        if n2 == 0; continue; end

                        # 同一モードから2つ選ぶ場合は、2個以上いる必要がある
                        if m1 == m2 && n2 < 2; continue; end
                        
                        # 散乱後のモード
                        m1_new = m1 + q
                        m2_new = m2 - q
                        if m1_new < 1 || m1_new > n_modes || m2_new < 1 || m2_new > n_modes
                            continue
                        end

                        transition_idx = base + t
                        # 状態をコピーして更新 (このスロットは以後上書きされない)
                        _update_proposed_states!(proposed_states, states, n_modes, m1, s, m2, s, m1_new, s, m2_new, s, transition_idx, w)

                        # 行列要素の計算
                        factor_annihilate = Float32(n1) * Float32(m1 == m2 ? n2 - 1 : n2)
                        matrix_elements[transition_idx, w] = _calculate_bose_factor(proposed_states, factor_annihilate, m1_new, s, m2_new, s, transition_idx, w)

                        t += 1
                    end
                end

                # 未使用スロット: 行列要素は0、提案状態には現在の状態を入れておく。
                while t <= max_t_per
                    transition_idx = base + t
                    matrix_elements[transition_idx, w] = 0.0f0
                    for ss in 1:3, m in 1:n_modes
                        proposed_states[m, ss, transition_idx, w] = states[m, ss, w]
                    end
                    t += 1
                end
            end
        end
    end

    return nothing
end


function compute_local_density_matrix(
    states::CuArray{Int32, 3}, 
    log_psi_current::CuArray{ComplexF32, 1}, 
    params::SystemParams, 
    threads::Int,
    model, ps, st
)
    n_walkers = size(states, 3)
    
    proposed_states = CUDA.zeros(Int32, 2 * params.k_max + 1, 3, params.n_modes^2, n_walkers)
    matrix_elements = CUDA.zeros(Float32, params.n_modes^2, n_walkers, 3)
 
    # (A) 遷移先状態 x' と、その行列要素 V_xx' を一括生成する関数
    # proposed_states: [n_modes, 3, max_transitions, n_walkers]
    # matrix_elements: [max_transitions, n_walkers, n_modes, 3]
    blocks = ceil(Int, n_walkers / threads)
    @cuda threads=threads blocks=blocks _density_matrix_local_estimator(
        states, 
        proposed_states, 
        matrix_elements, 
        params.k_max,
    )
    
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

    # 3. 合計して返す
    # １粒子密度行列 rho_kq を計算

    rho_kq = matrix_elements .* psi_ratio
    
    return rho_kq
end

function _density_matrix_local_estimator(
    states,             # [n_modes, 3, n_walkers] 現在の状態
    proposed_states,    # [n_modes, 3, MAX_TRANSITIONS, n_walkers] 遷移先を書き込むバッファ
    matrix_elements,    # [MAX_TRANSITIONS, n_walkers, 3] 行列要素 n_kk' を書き込むバッファ
    k_max::Int
)
    n_modes = 2 * k_max + 1
    w = (blockIdx().x - 1) * blockDim().x + threadIdx().x # 自分が担当するウォーカーID

    if w <= size(states, 3)
        
        # spin sの粒子数を計算
        for s in 1:3
            transition_idx = 1

            for m1 in 1:n_modes
                n1 = states[m1, s, w]
                if n1 == 0
                    matrix_elements[transition_idx, w, s] = 0.0f0
                    transition_idx += 1
                    continue
                end
                
                for m2 in m1+1:n_modes
                    n2 = states[m2, s, w]
                    if n2 == 0 
                        matrix_elements[transition_idx, w, s] = 0.0f0
                        transition_idx += 1
                        continue
                    end
                    
                    # 現在の運動量
                    k1 = m1 - k_max - 1
                    k2 = m2 - k_max - 1
                    
                    # 運動量移動 q のループ
                    # 状態をコピーして更新
                    for s in 1:3
                        for m in 1:n_modes
                            proposed_states[m, s, transition_idx, w] = states[m, s, w]
                        end
                    end
                    proposed_states[m1, s, transition_idx, w] -= 1
                    proposed_states[m2, s, transition_idx, w] += 1

                    # 行列要素の計算
                    bose_factor = sqrt(states[m1, s, w] * proposed_states[m2, s, transition_idx, w])
                    matrix_elements[transition_idx, w, s] = bose_factor

                    transition_idx += 1
                end
            end
        end
    end

    return nothing
end

end # module