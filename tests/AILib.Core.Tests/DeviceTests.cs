using Xunit;

namespace AILib.Tests;

public class DeviceTests
{
    [Fact]
    public void CPU_ReturnsValidDevice()
    {
        var device = Device.CPU;
        
        Assert.NotNull(device);
        Assert.Equal("cpu", device.Name);
    }

    [Fact]
    public void CPU_IsSingleton()
    {
        var device1 = Device.CPU;
        var device2 = Device.CPU;
        
        Assert.Same(device1, device2);
    }

    [Fact]
    public void Device_ToString_ReturnsName()
    {
        var device = Device.CPU;
        
        Assert.Equal("cpu", device.ToString());
    }
}
