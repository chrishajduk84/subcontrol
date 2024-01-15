#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/display.h>

#include <stdio.h>

const struct device* display_dev;

int main(void)
{
	printf("Starting %s Test Application\n", CONFIG_BOARD);
	display_dev = DEVICE_DT_GET(DT_CHOSEN(zephyr_display));
	return 0;
}
