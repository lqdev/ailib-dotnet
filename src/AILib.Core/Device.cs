namespace AILib;

/// <summary>
/// Represents a compute device where tensors reside.
/// Currently only CPU is supported.
/// </summary>
public abstract class Device
{
    /// <summary>
    /// Gets the singleton CPU device instance.
    /// </summary>
    public static CpuDevice CPU { get; } = CpuDevice.Instance;

    /// <summary>
    /// Gets the name of the device.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Returns a string representation of the device.
    /// </summary>
    public override string ToString() => Name;
}

/// <summary>
/// CPU device implementation.
/// </summary>
public sealed class CpuDevice : Device
{
    internal static readonly CpuDevice Instance = new();
    
    private CpuDevice() { }

    /// <inheritdoc/>
    public override string Name => "cpu";
}
