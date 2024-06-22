import smbus2
import time

# Define the BH1750 address
bh1750_address = 0x23

# Function to read data from the BH1750 sensor
def read_bh1750():
    with smbus2.SMBus(3) as bus:
        # Send measurement command to the sensor
        bus.write_byte(bh1750_address, 0x10)

        # Wait for measurement to complete
        time.sleep(0.5)

        # Read data from the sensor
        data = bus.read_i2c_block_data(bh1750_address, 0x00, 2)

        # Calculate the lux value from the received data
        lux = (data[1] + (256 * data[0])) / 1.2

    return lux

# Repeat the sensor reading
try:
    while True:
        # Read the sensor data
        lux = read_bh1750()

        # Print the lux value
        print(f'Lux: {lux:.2f}')

        # Delay between readings
        time.sleep(2)

except KeyboardInterrupt:
    pass