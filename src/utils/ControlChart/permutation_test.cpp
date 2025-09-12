#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <random>

namespace py = pybind11;

void randperm(int size, std::vector<int>& result, std::mt19937& rng) {
    std::iota(result.begin(), result.end(), 0);  // Fill with 0,1,2,...,size-1
    std::shuffle(result.begin(), result.end(), rng); 
}

bool permutationTest(const py::array_t<double>& rank, const py::array_t<double>& climit,
                     int m0, int num, int dim, int nbound, int minwin, int maxwin,
                     py::array_t<double>& Tval, const py::array_t<double>& mu,
                     const py::array_t<double>& sd, double lambda) {
    
    auto rank_unchecked = rank.unchecked<2>();
    auto climit_unchecked = climit.unchecked<1>();
    auto mu_unchecked = mu.unchecked<1>();
    auto sd_unchecked = sd.unchecked<1>();
    auto Tval_mutable = Tval.mutable_unchecked<1>();

    std::vector<int> sequence(num);
    std::vector<double> pRank(num * dim);
    std::vector<double> tmpRank(num * dim);

    std::random_device rd;
    std::mt19937 rng(rd());  // Seed with current time for randomness

    int win = std::min(std::max(num - m0, minwin), maxwin);
    int lower = std::max(m0, num - win);
    int icount = 0;

    while (icount < nbound) {
        randperm(num, sequence, rng);  // Properly shuffle sequence
        bool flag = true;

        for (int j = 0; j < dim; j++) {
            for (int i = 0; i < num; i++) {
                pRank[j * num + i] = rank_unchecked(sequence[i], j);  // Apply permuted indices
                tmpRank[j * num + i] = pRank[j * num + i];
            }
        }

        for (int i = 1; i < (num - lower); i++) {
            double pTval = 0;
            int pwin = std::min(std::max(num - m0 - i, minwin), maxwin);

            for (int j = 0; j < dim; j++) {
                double colTval = 0;
                for (int k = lower - pwin + 1; k < num - i; k++) {
                    if (pRank[j * num + k] > pRank[j * num + num - i]) {
                        tmpRank[j * num + k]--;
                    }
                    if (k >= num - i - pwin) {
                        colTval = (1 - lambda) * colTval + tmpRank[j * num + k] - (num - i + 1) / 2;
                    }
                }
                pTval += std::pow(colTval / sd_unchecked(num - m0 - i - 1), 2);
            }
            if (pTval >= climit_unchecked(num - m0 - i - 1)) {
                flag = false;
                break;
            }
        }

        if (flag) {
            double totalColTval = 0;
            for (int j = 0; j < dim; j++) {
                double colTval = 0;
                for (int k = num - win; k < num; k++) {
                    colTval = colTval * (1 - lambda) + pRank[j * num + k] - (num + 1) / 2;
                }
                totalColTval += std::pow(colTval / sd_unchecked(num - m0 - 1), 2);
            }
            Tval_mutable(icount++) = totalColTval;
        }
    }

    return true;
}


bool permutationTestDME(const py::array_t<double>& rank, const py::array_t<double>& climit,
    int m0, int num, int nbound, int minwin, int maxwin,
    py::array_t<double>& Tval, const py::array_t<double>& mu,
    const py::array_t<double>& sd, double lambda) {

    auto rank_unchecked = rank.unchecked<2>();
    auto climit_unchecked = climit.unchecked<1>();
    auto mu_unchecked = mu.unchecked<1>();
    auto sd_unchecked = sd.unchecked<1>();
    auto Tval_mutable = Tval.mutable_unchecked<1>();

    std::vector<int> sequence(num);
    std::vector<double> pRank(num);
    std::vector<double> tmpRank(num);

    std::random_device rd;
    std::mt19937 rng(rd());  // Random number generator

    int win = std::min(std::max(num - m0, minwin), maxwin);
    int lower = std::max(m0, num - win);
    int icount = 0;

    while (icount < nbound) {
        randperm(num, sequence, rng); 
        bool flag = true;

        for (int i = 0; i < num; i++) {
            pRank[i] = rank_unchecked(sequence[i], 0);
            tmpRank[i] = pRank[i];
        }

        for (int i = 1; i < (num - lower); i++) {
            double pTval = 0;
            int pwin = std::min(std::max(num - m0 - i, minwin), maxwin);
            double colTval = 0;
            for (int k = lower - pwin + 1; k < num - i; k++) {
                if (pRank[k] > pRank[num - i]) {        
                    tmpRank[k]--;
                }
                if (k >= num - i - pwin) {
                    colTval = (1 - lambda) * colTval + std::max(0.0, static_cast<double>(tmpRank[k] - ((num - i + 1) / 2.0)))-mu_unchecked(num - m0 - i - 1);
                }
            }
            pTval += std::pow(colTval / sd_unchecked(num - m0 - i - 1), 1);

            if (pTval >= climit_unchecked(num - m0 - i - 1)) {
                flag = false;
                break;
            }
        }

        if (flag) {
            double totalColTval = 0;
            double colTval = 0;
            for (int k = num - win; k < num; k++) {
                colTval = (1 - lambda) * colTval + std::max(0.0, static_cast<double>(pRank[k] - ((num + 1) / 2.0)))-mu_unchecked(num - m0 - 1);
            }
            totalColTval += std::pow(colTval / sd_unchecked(num - m0 - 1), 1);
            Tval_mutable(icount++) = totalColTval;
        }
    }
    return true;
}


PYBIND11_MODULE(permutation_test, m) {
    m.def("permutationTest", &permutationTest, 
          "Permutation test for sequential control charting",
          py::arg("rank"), py::arg("climit"), py::arg("m0"), py::arg("num"),
          py::arg("dim"), py::arg("nbound"), py::arg("minwin"), py::arg("maxwin"),
          py::arg("Tval"), py::arg("mu"), py::arg("sd"), py::arg("lambda"));
          
    m.def("permutationTestDME", &permutationTestDME, 
        "Permutation test for sequential control charting to detect positive mean shifts",
        py::arg("rank"), py::arg("climit"), py::arg("m0"), py::arg("num"),
        py::arg("nbound"), py::arg("minwin"), py::arg("maxwin"),
        py::arg("Tval"), py::arg("mu"), py::arg("sd"), py::arg("lambda"));
}
