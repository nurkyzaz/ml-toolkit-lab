

#include <iostream>
#include <cmath>

// Minimize f(x) = (x - 3)^2 via SGD (here deterministic gradient descent)
int main() {
    double x = 10.0;
    double lr = 0.1;

    for (int step = 1; step <= 50; step++) {
        // gradient of (x-3)^2 is 2(x-3)
        double grad = 2.0 * (x - 3.0);
        x -= lr * grad;

        double fx = (x - 3.0) * (x - 3.0);
        if (step % 10 == 0) {
            std::cout << "step " << step << " x=" << x << " f(x)=" << fx << "\n";
        }
    }

    std::cout << "final x=" << x << " (should be close to 3)\n";
    return 0;
}
