namespace AILib.NN.Loss;

/// <summary>
/// Mean Squared Error loss.
/// </summary>
public class MSELoss
{
    /// <summary>
    /// Computes MSE loss: mean((predictions - targets)²)
    /// </summary>
    /// <param name="predictions">Predicted values.</param>
    /// <param name="targets">Target values.</param>
    /// <returns>Scalar loss tensor.</returns>
    public Tensor<float> Forward(Tensor<float> predictions, Tensor<float> targets)
    {
        var diff = predictions - targets;
        var squared = diff * diff;
        return squared.Mean();
    }
}

/// <summary>
/// Binary Cross-Entropy loss.
/// </summary>
public class BCELoss
{
    private readonly float _eps;

    /// <summary>
    /// Initializes BCE loss.
    /// </summary>
    /// <param name="eps">Small constant to avoid log(0).</param>
    public BCELoss(float eps = 1e-7f)
    {
        _eps = eps;
    }

    /// <summary>
    /// Computes BCE loss: -mean(y*log(p) + (1-y)*log(1-p))
    /// </summary>
    /// <param name="predictions">Predicted probabilities (must be in [0, 1]).</param>
    /// <param name="targets">Target values (0 or 1).</param>
    /// <returns>Scalar loss tensor.</returns>
    public Tensor<float> Forward(Tensor<float> predictions, Tensor<float> targets)
    {
        // Clamp predictions to avoid log(0)
        var clampedPreds = predictions.Clamp(_eps, 1 - _eps);

        // BCE = -(y * log(p) + (1-y) * log(1-p))
        var term1 = targets * clampedPreds.Log();
        var oneMinusTargets = Tensor.Ones<float>(targets.Shape) - targets;
        var oneMinusPreds = Tensor.Ones<float>(predictions.Shape) - clampedPreds;
        var term2 = oneMinusTargets * oneMinusPreds.Log();

        var loss = -(term1 + term2);
        return loss.Mean();
    }
}

/// <summary>
/// Cross-Entropy loss for classification.
/// </summary>
public class CrossEntropyLoss
{
    /// <summary>
    /// Computes cross-entropy loss with softmax: -mean(log(softmax(logits)[target_class]))
    /// </summary>
    /// <param name="logits">Raw model outputs (before softmax) of shape [batch, num_classes].</param>
    /// <param name="targets">Target class indices of shape [batch].</param>
    /// <returns>Scalar loss tensor.</returns>
    public Tensor<float> Forward(Tensor<float> logits, Tensor<int> targets)
    {
        // For simplicity, we'll implement a basic version
        // A full implementation would use LogSoftmax for numerical stability

        // Compute softmax probabilities
        var probs = logits.Softmax(dim: -1);

        // For each sample, select the probability of the target class
        // This requires indexing which isn't fully implemented yet
        // For now, return a placeholder
        throw new NotImplementedException("CrossEntropyLoss with integer targets requires advanced indexing. Use BCELoss for binary classification.");
    }

    /// <summary>
    /// Computes cross-entropy loss with one-hot targets.
    /// </summary>
    /// <param name="logits">Raw model outputs of shape [batch, num_classes].</param>
    /// <param name="targets">One-hot encoded targets of shape [batch, num_classes].</param>
    /// <returns>Scalar loss tensor.</returns>
    public Tensor<float> ForwardOneHot(Tensor<float> logits, Tensor<float> targets)
    {
        // LogSoftmax for numerical stability
        var logSoftmax = logits.LogSoftmax();

        // Cross-entropy: -sum(targets * log_probs) / batch_size
        var loss = -(targets * logSoftmax).Sum();
        var batchSize = logits.Shape[0];
        return loss / (float)batchSize;
    }
}
