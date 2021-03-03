using CSV
using Plots

include("mlp.jl")

using .MLP

data = CSV.File("data/breast-cancer-wisconsin.data", header=false)
data = broadcast(collect.(data)) do x
    if x[7] == "?"
        x[7] = 0
    else
        x[7] = parse(Int, x[7])
    end
    x
end
data = collect(hcat(convert(Array{Array{Int64, 1}, 1}, data)...)')

d_tra = Dict(
    2 => [0, 1],
    4 => [1, 0],
)

data_X = data[:, 2:end-1] / 10
data_Y = collect(hcat(broadcast(data[:, end]) do x
    d_tra[x]
end...)')

data_X = Float64.(data_X)
data_Y = Float64.(data_Y)

let quz = "question-2", n_hidden_layers=5, n_epochs=800, η=0.9
    mlp = SimpleMLP{Float64}(data_X, data_Y, n_hidden_layers=n_hidden_layers, n_epochs=n_epochs, η=η)
    fit!(mlp)
    save_figures(mlp, "figs/$(quz)_")
    println("$(quz) done.")
end

let quz = "question-3", n_hidden_layers=5, η=0.9
    mlp = SimpleMLP{Float64}(data_X, data_Y, n_hidden_layers=n_hidden_layers, n_epochs=800, η=η)
    fit!(mlp)
    mlp_10 = SimpleMLP{Float64}(data_X, data_Y, n_hidden_layers=n_hidden_layers, n_epochs=10, η=η)
    fit!(mlp_10)
    mlp_100_000 = SimpleMLP{Float64}(data_X, data_Y, n_hidden_layers=n_hidden_layers, n_epochs=100_000, η=η)
    fit!(mlp_100_000)

    plt_1 = plot(
        mlp.loss_history,
        label="n_epochs=800, last=$(mlp.loss_history[end])",
        title="loss",
        legend=:outertopright,
        size=(1024, 512),
    )
    plot!(plt_1, mlp_10.loss_history, label="n_epochs=10, last=$(mlp_10.loss_history[end])")
    plot!(plt_1, mlp_100_000.loss_history, label="n_epochs=10,000, last=$(mlp_100_000.loss_history[end])")
    savefig(plt_1, "figs/$(quz)_loss_fig.png")

    plt_2 = plot(
        [mlp.train_accuracy_history mlp.test_accuracy_history],
        label=["trainset, n_epochs=800, last=$(mlp.train_accuracy_history[end])" "testset, n_epochs=800, last=$(mlp.test_accuracy_history[end])"],
        title="accuracy",
        legend=:outertopright,
        size=(1024, 512),
    )
    plot!(
        plt_2,
        [mlp_10.train_accuracy_history mlp_10.test_accuracy_history],
        label=["trainset, n_epochs=10, last=$(mlp_10.train_accuracy_history[end])" "testset, n_epochs=10, last=$(mlp_10.test_accuracy_history[end])"],
    )
    plot!(
        plt_2,
        [mlp_100_000.train_accuracy_history mlp_100_000.test_accuracy_history],
        label=["trainset, n_epochs=100,000, last=$(mlp_100_000.train_accuracy_history[end])" "testset, n_epochs=100,000, last=$(mlp_100_000.test_accuracy_history[end])"],
    )
    savefig(plt_2, "figs/$(quz)_accuracy_fig.png")
    println("$(quz) done.")
end

let quz = "question-4", n_epochs=800, η=0.9
    mlp = SimpleMLP{Float64}(data_X, data_Y, n_hidden_layers=5, n_epochs=n_epochs, η=η)
    fit!(mlp)
    mlp_2 = SimpleMLP{Float64}(data_X, data_Y, n_hidden_layers=2, n_epochs=n_epochs, η=η)
    fit!(mlp_2)
    mlp_30 = SimpleMLP{Float64}(data_X, data_Y, n_hidden_layers=30, n_epochs=n_epochs, η=η)
    fit!(mlp_30)

    plt_1 = plot(
        mlp.loss_history,
        label="n_hidden_layers=5, last=$(mlp.loss_history[end])",
        title="loss",
        legend=:outertopright,
        size=(1024, 512),
    )
    plot!(plt_1, mlp_2.loss_history, label="n_hidden_layers=2, last=$(mlp_2.loss_history[end])")
    plot!(plt_1, mlp_30.loss_history, label="n_hidden_layers=30, last=$(mlp_30.loss_history[end])")
    savefig(plt_1, "figs/$(quz)_loss_fig.png")

    plt_2 = plot(
        [mlp.train_accuracy_history mlp.test_accuracy_history],
        label=["trainset, n_hidden_layers=5, last=$(mlp.train_accuracy_history[end])" "testset, n_hidden_layers=5, last=$(mlp.test_accuracy_history[end])"],
        title="accuracy",
        legend=:outertopright,
        size=(1024, 512),
    )
    plot!(
        plt_2,
        [mlp_2.train_accuracy_history mlp_2.test_accuracy_history],
        label=["trainset, n_hidden_layers=2, last=$(mlp_2.train_accuracy_history[end])" "testset, n_hidden_layers=2, last=$(mlp_2.test_accuracy_history[end])"],
    )
    plot!(
        plt_2,
        [mlp_30.train_accuracy_history mlp_30.test_accuracy_history],
        label=["trainset, n_hidden_layers=30, last=$(mlp_30.train_accuracy_history[end])" "testset, n_hidden_layers=30, last=$(mlp_30.test_accuracy_history[end])"],
    )
    savefig(plt_2, "figs/$(quz)_accuracy_fig.png")
    println("$(quz) done.")
end

let quz = "question-5", n_tries=100, n_hidden_layers=5, n_epochs=800, η=0.9
    mlp_l = SimpleMLP{Float64}[]
    for t ∈ 1:n_tries
        mlp = SimpleMLP{Float64}(data_X, data_Y, n_hidden_layers=n_hidden_layers, n_epochs=n_epochs, η=η)
        fit!(mlp)
        push!(mlp_l, mlp)
    end
    w_mlp = mlp_l[argmax(map(x -> x.loss_history[end], mlp_l))]
    save_figures(w_mlp, "figs/$(quz)_")
    println("$(quz) done.")
end

let quz = "question-6", n_hidden_layers=5, n_epochs=800, η=0.9
    mlp = SimpleMLP{Float64}(data_X, data_Y, n_hidden_layers=n_hidden_layers, n_epochs=n_epochs, η=η)
    fit!(mlp)
    mlp_a = SimpleMLP{Float64}(data_X, data_Y, n_hidden_layers=n_hidden_layers, n_epochs=n_epochs, η=η)
    adaptive_fit!(mlp_a)

    plt_1 = plot(
        mlp.loss_history,
        label="Gradient Descent, last=$(mlp.loss_history[end])",
        title="loss",
        legend=:outertopright,
        size=(1024, 512),
    )
    plot!(plt_1, mlp_a.loss_history, label="AdaGrad, last=$(mlp_a.loss_history[end])")
    savefig(plt_1, "figs/$(quz)_loss_fig.png")

    plt_2 = plot(
        [mlp.train_accuracy_history mlp.test_accuracy_history],
        label=["trainset, Gradient Descent, last=$(mlp.train_accuracy_history[end])" "testset, Gradient Descent, last=$(mlp.test_accuracy_history[end])"],
        title="accuracy",
        legend=:outertopright,
        size=(1024, 512),
    )
    plot!(
        plt_2,
        [mlp_a.train_accuracy_history mlp_a.test_accuracy_history],
        label=["trainset, AdaGrad, last=$(mlp_a.train_accuracy_history[end])" "testset, AdaGrad, last=$(mlp_a.test_accuracy_history[end])"],
    )
    savefig(plt_2, "figs/$(quz)_accuracy_fig.png")
    println("$(quz) done.")
end
