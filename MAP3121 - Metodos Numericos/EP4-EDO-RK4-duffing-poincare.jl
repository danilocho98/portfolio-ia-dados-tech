# ============================================================
# EP4 - MAP0214 - Equações Diferenciais Ordinárias
# Linguagem: Julia
# Aluno: Danilo Cho (NUSP 10769448)
#
# Conteúdo:
#   I   - ODE 2ª ordem: Euler e RK4, comparação com solução exata
#   II1 - Duffing (poço duplo): diagramas de espaço de fase
#   II2 - Diagrama de bifurcação (secção de Poincaré)
#   II3 - Mapa de Poincaré
# ============================================================

using Plots

# ------------------------------------------------------------
# Funções utilitárias genéricas
# ------------------------------------------------------------

"""
    rk4_step!(g, t, y, z, h)

Dá UM passo de RK4 para uma EDO de 2ª ordem escrita como:

    y' =  z
    z' =  g(t, y, z)

Retorna (t_n+1, y_n+1, z_n+1).
"""
function rk4_step!(g, t, y, z, h)
    k1y = h * z
    k1z = h * g(t,           y,           z)

    k2y = h * (z + 0.5*k1z)
    k2z = h * g(t + 0.5*h,   y + 0.5*k1y, z + 0.5*k1z)

    k3y = h * (z + 0.5*k2z)
    k3z = h * g(t + 0.5*h,   y + 0.5*k2y, z + 0.5*k2z)

    k4y = h * (z + k3z)
    k4z = h * g(t + h,       y + k3y,     z + k3z)

    y_new = y + (k1y + 2k2y + 2k3y + k4y)/6
    z_new = z + (k1z + 2k2z + 2k3z + k4z)/6
    t_new = t + h

    return t_new, y_new, z_new
end

# ============================================================
# I) y'' = y' + y - t^3 - 4 t^2 + 4 t + 2 , y(0)=0, y'(0)=0
#     g(t,y,z) = z + y - t^3 - 4 t^2 + 4 t + 2
#     solução exata: y(t) = t^3 + t^2
# ============================================================

# RHS g(t,y,z) da equação do enunciado
g_ode(t, y, z) = z + y - t^3 - 4t^2 + 4t + 2

# solução exata
y_exata(t) = t^3 + t^2
z_exata(t) = 3t^2 + 2t       # y'(t)

"""
    resolver_parte_I(h)

Resolve a ODE de 2ª ordem por Euler e RK4 no intervalo [0,6]
com passo h, e gera o gráfico comparando com y_exata.
"""
function resolver_parte_I(h::Float64)
    t0, tf = 0.0, 6.0
    N = Int(round((tf - t0)/h))

    # vetores
    t  = range(t0, length = N+1, step = h)
    yE = zeros(N+1)
    zE = zeros(N+1)

    yR = zeros(N+1)
    zR = zeros(N+1)

    # condições iniciais
    yE[1] = 0.0; zE[1] = 0.0
    yR[1] = 0.0; zR[1] = 0.0

    # -------- Euler --------
    for n in 1:N
        yE[n+1] = yE[n] + h * zE[n]
        zE[n+1] = zE[n] + h * g_ode(t[n], yE[n], zE[n])
    end

    # -------- RK4 --------
    t_rk = t0
    y = 0.0
    z = 0.0
    yR[1] = y
    zR[1] = z
    for n in 1:N
        t_rk, y, z = rk4_step!(g_ode, t_rk, y, z, h)
        yR[n+1] = y
        zR[n+1] = z
    end

    # valores em t=6
    yE6, zE6 = yE[end], zE[end]
    yR6, zR6 = yR[end], zR[end]
    yex6, zex6 = y_exata(tf), z_exata(tf)

    println("============== PARTE I – RESULTADOS EM t = 6 =================")
    println("Passo h = $(h)")
    println("Solução exata:  y(6) = $(yex6),  y'(6) = $(zex6)")
    println("Euler:          y(6) = $(yE6),  y'(6) = $(zE6)")
    println("RK4:            y(6) = $(yR6),  y'(6) = $(zR6)")
    println("Erro Euler: |y-y_ex| = $(abs(yE6-yex6)),  |y'-y'_ex| = $(abs(zE6-zex6))")
    println("Erro RK4:   |y-y_ex| = $(abs(yR6-yex6)),  |y'-y'_ex| = $(abs(zR6-zex6))")
    println("================================================================\n")

    # ---- Gráfico y(t): Euler, RK4 e exato ----
    y_ex = [y_exata(tt) for tt in t]

    plt = plot(t, y_ex, label = "Exata", lw=2, color=:black)
    plot!(plt, t, yE, label = "Euler", lw=2, ls=:dash)
    plot!(plt, t, yR, label = "RK4", lw=2, ls=:dot)
    xlabel!("t")
    ylabel!("y(t)")
    title!("Parte I – y(t): Exata x Euler x RK4")
    grid!(true)
    display(plt)

    return nothing
end

# ============================================================
# II) Equação de Duffing / Poço duplo
#
#    Forma geral usada:
#      x' = v
#      v' = -2γ v + 0.5 x (1 - x^2) + F*cos(ω t)
#
#    Casos:
#      (a) γ = 0, F = 0
#      (b) γ > 0, F = 0
#      (c) γ = 0.125 (2γ = 0.25), F ≠ 0, ω = 0.95
# ============================================================

# RHS do sistema de Duffing / poço duplo
function g_duffing(t, x, v; γ=0.0, F=0.0, ω=0.95)
    return -2γ*v + 0.5*x*(1 - x^2) + F*cos(ω*t)
end

"""
    simular_duffing!(tmax; h, x0, v0, γ, F, ω)

Evolui o sistema de Duffing até tmax usando RK4 (passo h).
Retorna (t_vec, x_vec, v_vec).
"""
function simular_duffing(tmax; h=0.01, x0=-1.0, v0=0.0,
                         γ=0.0, F=0.0, ω=0.95)
    N = Int(round(tmax / h))
    t  = range(0.0, length=N+1, step=h)
    x  = zeros(N+1)
    v  = zeros(N+1)
    x[1] = x0
    v[1] = v0

    tcur = 0.0
    xcur = x0
    vcur = v0

    for n in 1:N
        g_loc(t,y,z) = g_duffing(t,y,z; γ=γ, F=F, ω=ω)
        tcur, xcur, vcur = rk4_step!(g_loc, tcur, xcur, vcur, h)
        x[n+1] = xcur
        v[n+1] = vcur
    end

    return t, x, v
end

# ------------------ II-1a: sem amortecimento e sem força -------------------

function parte_II_1a()
    println("PARTE II-1a) Duffing sem amortecimento e sem força (γ=0, F=0)")

    h = 0.01
    tmax = 200.0    # pode ajustar se quiser mais/menos voltas
    x0 = -1.0

    v0_list = [0.1, 0.5, 1.0]

    plt = plot(title = "Espaço de fase – Duffing (γ=0, F=0)",
               xlabel = "x(t)", ylabel = "v(t) = ẋ(t)",
               legend = :bottomright)

    for v0 in v0_list
        t, x, v = simular_duffing(tmax; h=h, x0=x0, v0=v0, γ=0.0, F=0.0)
        plot!(plt, x, v, label = "v₀ = $(v0)")
    end

    grid!(plt, true)
    display(plt)
    return nothing
end

# ------------------ II-1b: com amortecimento -------------------

function parte_II_1b()
    println("PARTE II-1b) Duffing com amortecimento (2γ = 0.25 e 0.8)")

    h = 0.01
    tmax = 200.0
    x0 = -1.0
    v0 = 1.0

    γ_list = [0.25/2, 0.8/2]  # pois equação é x¨ + 2γ x˙ - ... = 0

    plt = plot(title = "Espaço de fase – Duffing amortecido (F=0)",
               xlabel = "x(t)", ylabel = "v(t)",
               legend = :bottomright)

    for γ in γ_list
        t, x, v = simular_duffing(tmax; h=h, x0=x0, v0=v0, γ=γ, F=0.0)
        plot!(plt, x, v, label = "2γ = $(2γ)")
    end

    grid!(plt, true)
    display(plt)
    return nothing
end

# ------------------ II-1c: com força periódica -------------------

function parte_II_1c()
    println("PARTE II-1c) Duffing forçado: x¨ + 0.25 ẋ - 0.5 x(1-x²) = F cos(ω t)")

    ω = 0.95
    γ = 0.25/2          # porque x¨ + 2γ x˙ - 0.5x(1-x²) = F cos(ωt)
    F_list = [0.19, 0.203, 0.24, 0.33, 0.6]

    h = 0.01 * 2π/ω     # passo ~ 1% do período
    tmax = 600.0        # tempo longo para ver o atrator
    x0 = -1.0
    v0 = 1.0

    for F in F_list
        println("  Evoluindo para F = $F ... (removendo transitório)")

        t, x, v = simular_duffing(tmax; h=h, x0=x0, v0=v0, γ=γ, F=F, ω=ω)

        # Remove um pedaço inicial como "transiente"
        ncut = Int(round(0.5 * length(t)))  # descarta 50% inicial
        x_plot = x[ncut:end]
        v_plot = v[ncut:end]

        plt = plot(x_plot, v_plot,
                   xlabel = "x(t)", ylabel = "v(t)",
                   title = "Espaço de fase – Duffing forçado, F = $(F)",
                   legend = false)
        grid!(plt, true)
        display(plt)
    end

    println("\nPergunta II-1d: Os atratores observados são:")
    println("- Caso (a): órbitas fechadas (poço duplo conservativo) → órbitas periódicas em torno dos mínimos.")
    println("- Caso (b): com amortecimento → as trajetórias espiralam até pontos fixos (atratores de ponto).")
    println("- Caso (c): com força periódica → órbitas periódicas, quase-periódicas ou caóticas, dependendo de F.")

    return nothing
end

# ============================================================
# II-2) Diagrama de bifurcação (secções de Poincaré)
# F: 0 → 0.7, passo ΔF = 0.0005
# ============================================================

function parte_II_2(; Fmin=0.0, Fmax=0.7, dF=0.0005, ω=0.95)
    println("PARTE II-2) Diagrama de bifurcação (isso pode demorar alguns minutos)...")

    γ = 0.25/2
    Fs = Float64[]
    Xs = Float64[]

    # passos sugeridos
    h_trans = 0.01  * 2π/ω
    h       = 0.001 * 2π/ω

    for F in Fmin:dF:Fmax
        t = 0.0
        x = -1.0
        v = 1.0

        g_loc(t,y,z) = g_duffing(t,y,z; γ=γ, F=F, ω=ω)

        # Transiente: 200000 passos (anda muitos períodos)
        for n in 1:200_000
            t, x, v = rk4_step!(g_loc, t, x, v, h_trans)
        end

        # Agora faz 100 períodos; cada período ~ 2π/ω,
        # com 1000 passos → 1000 h = 2π/ω
        for i in 1:100
            for j in 1:1000
                t, x, v = rk4_step!(g_loc, t, x, v, h)
            end
            push!(Fs, F)
            push!(Xs, x)  # ponto da secção de Poincaré: x em t ≈ múltiplos de 2π/ω
        end
    end

    plt = scatter(Fs, Xs; markersize=1, legend=false)
    xlabel!("F")
    ylabel!("x (secção de Poincaré)")
    title!("Diagrama de bifurcação – Duffing forçado")
    grid!(true)
    display(plt)

    println("\nA partir deste diagrama, pode-se estimar manualmente a constante de Feigenbaum δ")
    println("medindo as distâncias entre sucessivas bifurcações em F.")
    return nothing
end

# ============================================================
# II-3) Mapa de Poincaré fixando F = 0.24
# ============================================================

function parte_II_3(; F=0.24, ω=0.95)
    println("PARTE II-3) Mapa de Poincaré em F = $F")

    γ = 0.25/2
    h_trans = 0.01  * 2π/ω
    h       = 0.001 * 2π/ω

    # condições iniciais
    t = 0.0
    x = -1.0
    v = 1.0

    g_loc(t,y,z) = g_duffing(t,y,z; γ=γ, F=F, ω=ω)

    # Transiente longo
    for n in 1:200_000
        t, x, v = rk4_step!(g_loc, t, x, v, h_trans)
    end

    # Agora fazemos i = 1..20000 pontos na secção (Poincaré)
    Xp = Float64[]
    Vp = Float64[]

    for i in 1:20_000
        for j in 1:1000
            t, x, v = rk4_step!(g_loc, t, x, v, h)
        end
        push!(Xp, x)
        push!(Vp, v)
    end

    plt = scatter(Xp, Vp; markersize=1, legend=false)
    xlabel!("x")
    ylabel!("v = ẋ")
    title!("Mapa de Poincaré – Duffing forçado, F = $F")
    grid!(true)
    display(plt)

    return nothing
end

# ============================================================
# "Main" – chame aqui o que você quiser rodar
# ============================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # ---- Parte I: Euler x RK4 na ODE de 2ª ordem ----
    resolver_parte_I(0.01)   # h = 0.01

    # ---- Parte II-1: Espaço de fase ----
    parte_II_1a()
    parte_II_1b()
    parte_II_1c()

    # ---- Parte II-2: Diagrama de bifurcação ----
    # (pode demorar alguns minutos; se estiver testando, pode diminuir o range de F)
    # parte_II_2()

    # ---- Parte II-3: Mapa de Poincaré ----
    # parte_II_3()
end
