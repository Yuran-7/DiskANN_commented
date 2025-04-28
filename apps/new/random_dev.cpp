#include <iostream>
#include <random>

int main() {
    std::random_device rand_dev;

    // 每次运行程序时，输出的值通常不同
    std::cout << "Random number: " << rand_dev() << std::endl;

    return 0;
}