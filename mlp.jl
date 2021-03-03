module MLP
    using Random
    using Statistics
    using Plots
    using ProgressMeter

    export
        SimpleMLP,
        fit!,
        adaptive_fit!,
        predict,
        save_figures

    abstract type NeuralNetwork end
    abstract type FeedForwardNN <: NeuralNetwork end
    abstract type MultiLayerPerceptron <: FeedForwardNN end

    error(ŷ, y) = ŷ - y
    σ(x) = 1 / (1 + exp(-x))
    σ′(x) = x * (1 - x)

    mutable struct SimpleMLP{T} <: MultiLayerPerceptron where T <: AbstractFloat
        train_data
        test_data

        loss
        activation
        activation_gradient
        n_epochs::Integer
        η::T

        wL₁::Matrix{T}
        wL₂::Matrix{T}
        loss_history::Vector{T}
        train_accuracy_history::Vector{T}
        test_accuracy_history::Vector{T}

        function SimpleMLP{T}(
                data_X::Matrix{T},
                data_Y::Matrix{T},
                ;
                train_test_ratio::AbstractFloat=0.8,
                loss=error,
                activation=σ,
                activation_gradient=σ′,
                n_hidden_layers::Integer=size(data_X, 2),
                n_epochs::Integer=100,
                η::T=convert(T, 1e-1),
        ) where T <: AbstractFloat
            data = shuffle(collect(zip(eachrow(data_X), eachrow(data_Y))))
            n_train_data = round(Integer, length(data) * train_test_ratio)
            new(data[1:n_train_data], data[n_train_data+1:end],
                loss, activation, activation_gradient, n_epochs, η,
                rand(n_hidden_layers, size(data_X, 2)), rand(size(data_Y, 2), n_hidden_layers), T[], T[], T[])
        end
    end

    function fit!(mlp::MultiLayerPerceptron)
        last_up_wL₂ = zeros(size(mlp.wL₂))
        new_up_wL₂ = zeros(size(mlp.wL₂))
        last_up_wL₁ = zeros(size(mlp.wL₁))
        new_up_wL₁ = zeros(size(mlp.wL₁))
        @showprogress for c ∈ 1:mlp.n_epochs
            for (x, y) ∈ mlp.train_data
                x = hcat(x)
                y = hcat(y)
                aL₁ = mlp.activation.(mlp.wL₁ * x)
                aL₂ = mlp.activation.(mlp.wL₂ * aL₁)
                e_nn = mlp.loss(aL₂, y)
                dL₂ = mlp.activation_gradient.(aL₂) .* e_nn
                dL₁ = mlp.activation_gradient.(aL₁) .* (mlp.wL₂' * dL₂)

                ΔL₂ = dL₂ * aL₁'
                ΔL₁ = dL₁ * x'
                ΔL₂ *= mlp.η
                ΔL₁ *= mlp.η
                mlp.wL₂ -= ΔL₂
                mlp.wL₁ -= ΔL₁
                push!(mlp.loss_history, mean(e_nn))
            end
            check_accuracy!(mlp)
        end
    end

    function adaptive_fit!(mlp::MultiLayerPerceptron)
        ϵ = eps()
        GwL₂ = zeros(size(mlp.wL₂))
        GwL₁ = zeros(size(mlp.wL₁))
        @showprogress for c ∈ 1:mlp.n_epochs
            for (x, y) ∈ mlp.train_data
                x = hcat(x)
                y = hcat(y)
                aL₁ = mlp.activation.(mlp.wL₁ * x)
                aL₂ = mlp.activation.(mlp.wL₂ * aL₁)
                e_nn = mlp.loss(aL₂, y)
                dL₂ = mlp.activation_gradient.(aL₂) .* e_nn
                dL₁ = mlp.activation_gradient.(aL₁) .* (mlp.wL₂' * dL₂)

                ΔL₂ = dL₂ * aL₁'
                ΔL₁ = dL₁ * x'
                GwL₂ += ΔL₂ .^ 2
                GwL₁ += ΔL₁ .^ 2
                @. ΔL₂ *= mlp.η / (√GwL₂ + ϵ)
                @. ΔL₁ *= mlp.η / (√GwL₁ + ϵ)
                mlp.wL₂ -= ΔL₂
                mlp.wL₁ -= ΔL₁
                push!(mlp.loss_history, mean(e_nn))
            end
            check_accuracy!(mlp)
        end
    end

    function predict(mlp::MultiLayerPerceptron, x::Matrix)
        mlp.activation.(mlp.wL₂ * mlp.activation.(mlp.wL₁ * x))
    end

    function save_figures(mlp::MultiLayerPerceptron, fn::AbstractString="figs/")
        savefig(plot(
            mlp.loss_history,
            label="loss",
            title="loss",
            legend=:outertopright,
            size=(1024, 512),
        ), "$(fn)loss_fig.png")
        savefig(plot(
            [mlp.train_accuracy_history mlp.test_accuracy_history],
            label=["train accuracy" "test accuracy"],
            title="accuracy",
            legend=:outertopright,
            size=(1024, 512),
        ), "$(fn)accuracy_fig.png")
    end

    function check_accuracy!(mlp::MultiLayerPerceptron)
        train_true_prediction = 0
        for (x, y) ∈ mlp.train_data
            x = hcat(x)
            y = hcat(y)
            out = predict(mlp, x)
            train_true_prediction += argmax(out) == argmax(y)
        end
        train_acc = train_true_prediction / length(mlp.train_data)
        push!(mlp.train_accuracy_history, train_acc)

        test_true_prediction = 0
        for (x, y) ∈ mlp.test_data
            x = hcat(x)
            y = hcat(y)
            out = predict(mlp, x)
            test_true_prediction += argmax(out) == argmax(y)
        end
        test_acc = test_true_prediction / length(mlp.test_data)
        push!(mlp.test_accuracy_history, test_acc)
    end
end