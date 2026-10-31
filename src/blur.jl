@inline function _nchan(s1, s2)
    (s1 == 2 && s2 == 2) && return 3
    ((s1 == 1 && s2 == 3) || (s1 == 3 && s2 == 1)) && return 2
    abs(s1 - s2) == 1 && return 2
    return 1
end
@inline function _chan(s1, s2, j)             # j-th spin channel, same order as _scattering_kernel!
    j == 1 && return (s1, s2)
    (s1 == 2 && s2 == 2) && return j == 2 ? (1, 3) : (3, 1)
    ((s1 == 1 && s2 == 3) || (s1 == 3 && s2 == 1)) && return (2, 2)
    return (s2, s1)
end

# T[b] = 候補リストの長さ（_scattering_kernel! が書く遷移数と一致）
function _count_transitions!(T, S, k_max::Int)   # S :: [n_modes, 3, B]
    n_modes = 2k_max + 1
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if b <= size(S, 3)
        t = Int32(0)
        for m1 in 1:n_modes, s1 in 1:3
            S[m1, s1, b] == 0 && continue
            for m2 in 1:n_modes, s2 in 1:3
                n2 = S[m2, s2, b]
                n2 == 0 && continue
                (m1 == m2 && s1 == s2 && n2 < 2) && continue
                t += Int32((n_modes - abs(m1 + m2 - 2(k_max + 1))) * _nchan(s1, s2))
            end
        end
        T[b] = t
    end
    return nothing
end

# 確率 q で、x の候補リストの一様な1項を直接適用（全列挙しない）
function _pick_blurred!(Y, X, Tx, R, q::Float32, k_max::Int)
    n_modes = 2k_max + 1
    w = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if w <= size(X, 3)
        for s in 1:3, m in 1:n_modes
            Y[m, s, w] = X[m, s, w]
        end
        (R[1, w] >= q || Tx[w] == 0) && return nothing
        target = min(trunc(Int32, R[2, w] * Tx[w]) + Int32(1), Tx[w])
        acc = Int32(0)
        for m1 in 1:n_modes, s1 in 1:3
            X[m1, s1, w] == 0 && continue
            for m2 in 1:n_modes, s2 in 1:3
                n2 = X[m2, s2, w]
                n2 == 0 && continue
                (m1 == m2 && s1 == s2 && n2 < 2) && continue
                k1 = m1 - k_max - 1; k2 = m2 - k_max - 1
                nc = _nchan(s1, s2)
                cnt = Int32((n_modes - abs(k1 + k2)) * nc)
                if target <= acc + cnt
                    loc = target - acc - 1
                    dq  = max(-k_max - k1, k2 - k_max) + loc ÷ nc
                    a, b = _chan(s1, s2, loc % nc + 1)
                    Y[m1, s1, w] -= 1; Y[m2, s2, w] -= 1
                    Y[k1 + dq + k_max + 1, a, w] += 1
                    Y[k2 - dq + k_max + 1, b, w] += 1
                    return nothing
                end
                acc += cnt
            end
        end
    end
    return nothing
end

"""
戻り値: (blurred_states, ω, proposed_states, n_conn, log_psi_real_nb)
proposed_states は x' の近傍なので、そのまま x' での局所エネルギー計算に使える。
"""
function blur(states::CuArray{Float32,3}, proposed_states::CuArray{Float32,4},
              params::SystemParams, q::Float32, threads::Int, model, ps, st)
    B = size(states, 3); MAXT = size(proposed_states, 3)
    k_max = params.k_max; n_modes = params.n_modes
    blocks = cld(B, threads)

    # 1. x の遷移数 → t 番目の遷移を直接適用して x'
    Tx = CUDA.zeros(Int32, B)
    @cuda threads=threads blocks=blocks _count_transitions!(Tx, states, k_max)
    blurred = similar(states)
    R = CUDA.rand(Float32, 2, B)
    @cuda threads=threads blocks=blocks _pick_blurred!(blurred, states, Tx, R, q, k_max)

    # 2. x' の近傍を列挙（局所エネルギーと共用）。n_conn = T(x')
    fill!(proposed_states, 0f0)
    @cuda threads=threads blocks=blocks _scattering_kernel!(blurred, proposed_states, k_max)
    n_conn = CUDA.zeros(Int32, B)
    @cuda threads=threads blocks=blocks _count_transitions!(n_conn, blurred, k_max)
    @assert maximum(n_conn) <= MAXT "MAX_TRANSITIONS が不足"

    # 3. 各近傍 y の T(y)
    flat = reshape(proposed_states, n_modes, 3, :)
    Tnb = CUDA.zeros(Int32, size(flat, 3))
    @cuda threads=threads blocks=cld(size(flat, 3), threads) _count_transitions!(Tnb, flat, k_max)
    Tnb = reshape(Tnb, MAXT, B)

    # 4. 振幅（実部のみで足りる）
    lp0 = eval_complex_network_real(model, blurred, ps, st)
    lpn = reshape(eval_complex_network_real(model, flat, ps, st), MAXT, B)

    # 5. ω = 1 / [(1-q) + q Σ_{リスト全項} |ψ(y)/ψ(x')|² / T(y)]   （自己ループ込み）
    valid = reshape(1:MAXT, :, 1) .<= reshape(n_conn, 1, :)
    d   = ifelse.(valid, 2f0 .* (lpn .- reshape(lp0, 1, :)), -Inf32)   # exp の前でマスク
    trm = ifelse.(valid, exp.(d) ./ max.(Tnb, Int32(1)), 0f0)
    w   = 1f0 ./ ((1 - q) .+ q .* vec(sum(trm; dims = 1)))           # Inf → ω=0 で正しい極限

    return blurred, w
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



