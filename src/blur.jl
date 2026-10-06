function blur(
    states::CuArray{Float32, 3}, 
    proposed_states::CuArray{Float32, 4},
    params::SystemParams, 
    q::Float32, 
    threads::Int,
    model, ps, st
)   
    n_walkers = size(states, 3)

    ## 散乱カーネルを用いて提案状態候補を作成
    proposed_states .= CUDA.zeros(Float32, size(proposed_states))
    blocks = ceil(Int, n_walkers / threads)
    @cuda threads=threads blocks=blocks _scattering_kernel!(
        states, 
        proposed_states, 
        params.k_max,
    )

    ## 提案状態候補のインデックス・要素数を計算
    indices = CUDA.zeros(Int32, size(proposed_states, 3), n_walkers)
    n_conn = CUDA.zeros(Int32, n_walkers)
    @cuda threads=threads blocks=blocks _get_connected_indices!(
        indices, 
        n_conn, 
        proposed_states,
        params.k_max
    )

    ## 候補から一様ランダムに一つ選択
    blurred_states = CUDA.zeros(Float32, size(states))
    blurred_states .= states
    rand_vals = CUDA.rand(Float32, 2, n_walkers)
    @cuda threads=threads blocks=blocks _proposed_blurred_states!(
        blurred_states, 
        proposed_states, 
        indices, 
        n_conn, 
        rand_vals, 
        q, params.k_max
    )
 
    ## wを計算
    log_psi_real_prop = eval_complex_network_real(model, blurred_states, ps, st)

    flat_blurred = reshape(proposed_states, params.n_modes, 3, :)
    flat_log_psi_real_blurred = eval_complex_network_real(model, flat_blurred, ps, st)
    log_psi_real_blurred = reshape(flat_log_psi_real_blurred, size(proposed_states, 3), :)

    w_deno = (1 - q) .+ q ./ n_conn .* dropdims(sum(exp.(2 .* (log_psi_real_blurred .- reshape(log_psi_real_prop, 1, :))) .* (indices .> 0), dims=1), dims=1)

    return blurred_states, 1 ./ w_deno
end

"""
全ウォーカーのすべての2体散乱プロセスを列挙し、遷移先状態と行列要素を書き込むカーネル
"""
function _scattering_kernel!(
    states,             # [n_modes, 3, n_walkers] 現在の状態
    proposed_states,    # [n_modes, 3, MAX_TRANSITIONS, n_walkers] 遷移先を書き込むバッファ
    k_max::Int
)
    n_modes = 2 * k_max + 1
    w = (blockIdx().x - 1) * blockDim().x + threadIdx().x # 自分が担当するウォーカーID
    
    if w <= size(states, 3)
        transition_idx = 1
        
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
                
                # 運動量移動 q のループ
                for q in -2*k_max:2*k_max
                    # 散乱後の運動量
                    k1_new = k1 + q
                    k2_new = k2 - q

                    m1_new = k1_new + k_max + 1
                    m2_new = k2_new + k_max + 1

                    if m1_new >= 1 && m1_new <= n_modes && m2_new >= 1 && m2_new <= n_modes
                                               
                        # 状態をコピーして更新
                        _update_proposed_states!(proposed_states, states, n_modes, m1, s1, m2, s2, m1_new, s1, m2_new, s2, transition_idx, w)
                        
                        transition_idx += 1

                        # (0, 0) <--> (1, -1) の遷移
                        if s1 == 2 && s2 == 2
                            s1_new, s2_new = 1, 3
                            
                            # 状態をコピーして更新
                            _update_proposed_states!(proposed_states, states, n_modes, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, transition_idx, w)

                            transition_idx += 1

                            s1_new, s2_new = 3, 1
                            
                            # 状態をコピーして更新
                            _update_proposed_states!(proposed_states, states, n_modes, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, transition_idx, w)

                            transition_idx += 1
                            
                        # (1, -1) <--> (0, 0), (-1, 1) <--> (0, 0) の遷移
                        elseif (s1 == 1 && s2 == 3) || (s1 == 3 && s2 == 1)
                            s1_new, s2_new = 2, 2
                            
                            # 状態をコピーして更新
                            _update_proposed_states!(proposed_states, states, n_modes, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, transition_idx, w)
                            
                            transition_idx += 1
                            
                        # スピンの交換
                        elseif abs(s1 - s2) == 1
                            s1_new, s2_new = s2, s1
                            
                            # 状態をコピーして更新
                            _update_proposed_states!(proposed_states, states, n_modes, m1, s1, m2, s2, m1_new, s1_new, m2_new, s2_new, transition_idx, w)

                            transition_idx += 1
                        end
                    end
                end
            end
        end
    end
    
    return nothing
end

function _get_connected_indices!(indices, n_conn, proposed_states, k_max::Int)
    n_modes = 2 * k_max + 1
    w = (blockIdx().x - 1) * blockDim().x + threadIdx().x # 自分が担当するウォーカーID
    if w <= size(proposed_states, 3)
        n = 0

        for t in axes(proposed_states, 3)
            n_particles = 0
            for m in 1:n_modes, s in 1:3
                n_particles += proposed_states[m, s, t, w]
            end
            if n_particles == 0 continue end
    
            indices[n + 1, w] = t
            n += 1
        end
        n_conn[w] = n
    end

    return nothing
end
        
function _proposed_blurred_states!(blurred_states, proposed_states, indices, n_conn, rand_vals, q::Float32, k_max::Int)
    n_modes = 2 * k_max + 1
    w = (blockIdx().x - 1) * blockDim().x + threadIdx().x # 自分が担当するウォーカーID
    if w <= size(blurred_states, 3)
        n = n_conn[w]

        ## 非対角カーネルで更新
        if rand_vals[2, w] < q
            ## 非対角成分を一様に選択
            selected_idx = min(trunc(Int32, rand_vals[1, w] * n) + 1, n)
            index = indices[selected_idx, w]
            for s in 1:3
                for m in 1:n_modes
                    blurred_states[m, s, w] = proposed_states[m, s, index, w]
                end
            end
        end
    end
    
    return nothing
end



