// benchmark_models.cpp (Reads discrete dividend adjusted input)

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>    // For erf, sqrt, log, pow, isnan, abs, exp, M_SQRT1_2 (maybe)
#include <chrono>   // For time points and duration
#include <map>
#include <limits>   // For numeric_limits
#include <iomanip>  // For setting precision
#include <stdexcept> // For exceptions
#include <algorithm> // For std::max, std::min
#include <numeric>   // For std::accumulate, std::inner_product
#include <functional> // For std::function
#include <memory>    // For std::unique_ptr
#include <ctime>     // For time_t, tm, timegm/_mkgmtime, gmtime_s/gmtime
#include <cctype>    // For std::isdigit

// --- Hardcoded Settings ---
const int BENCHMARK_N_RUNS = 100; // Match Python settings.py if needed
const int BINOMIAL_N_STEPS = 101; // Match Python settings.py if needed
// *** Hardcoded Paths ***
const std::string INPUT_DIR = "pipeline_output_final/benchmark_input_data"; // Dir containing input CSV
const std::string OUTPUT_DIR = "pipeline_output_final/graph_data";    // Dir to save C++ results
// ------------------------------------

// --- Option Type Enum & BenchmarkResult Struct ---
enum class OptionType { CALL, PUT };
// Result struct matches the desired output CSV columns
struct BenchmarkResult {
    std::string underlying_ticker; std::string option_ticker; int target_dte_group; std::string option_type; std::string timestamp;
    std::string model; int binomial_steps; int run_number; double calc_time_sec; double calculated_price;
    double delta; double gamma; double vega; double theta; double rho;
    double input_S; double input_K; double input_T; double input_r; double input_sigma; double input_q;
};

// --- Utility Functions ---
double normalCDF(double x) {
    #ifndef M_SQRT1_2
    const double M_SQRT1_2 = 0.70710678118654752440; 
    #endif
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

double bisection(std::function<double(double)> func, double a, double b, double tol = 1e-7, int max_iter = 100) { if (func(a) * func(b) >= 0) { if (std::abs(func(a)) < tol) return a; if (std::abs(func(b)) < tol) return b; return std::numeric_limits<double>::quiet_NaN(); } double c = a; for (int i = 0; i < max_iter; ++i) { c = a + (b - a) / 2.0; if (std::abs(func(c)) < tol || (b - a) / 2.0 < tol) { return c; } if (func(c) * func(a) < 0) { b = c; } else { a = c; } } return c; }

// --- Base Class and Derived Classes for American Option Pricing ---
lass AmericanOptionPricerBase {
public:
    double S, K, T, r, sigma, q;
    OptionType option_type;
    double _dS_ratio = 0.001;
    double _dT_years;
    double _dSigma = 0.0005;
    double _dr = 0.00005;
    int N_steps_member = 0; // Used by derived classes

    AmericanOptionPricerBase(double s, double k, double t, double r_, double sigma_, double q_, OptionType type) :
        S(s),
        K(k),
        T(std::max(t, 1e-9)),
        r(r_),
        sigma(std::max(sigma_, 1e-9)),
        q(q_),
        option_type(type)
    {
        _dT_years = std::min(0.002, T * 0.5);
        if (option_type != OptionType::CALL && option_type != OptionType::PUT) {
            throw std::invalid_argument("option_type must be OptionType::CALL or OptionType::PUT");
        }
    }

    virtual ~AmericanOptionPricerBase() = default;

    virtual double _price_impl() = 0;

    virtual std::map<std::string, double> _greeks_impl() {
        double price_mid = this->_price_impl(); // Calculate central price first
        double orig_S = S, orig_T = T, orig_sigma = sigma, orig_r = r; // Store original params
        double dS = orig_S * _dS_ratio;

        double delta = NAN, gamma = NAN, vega = NAN, theta = NAN, rho = NAN; // Initialize to NAN

        try {
            // Delta and Gamma
            S = orig_S + dS;
            double price_up_S = this->_price_impl();
            S = orig_S - dS;
            double price_down_S = this->_price_impl();
            S = orig_S; // Restore S
            if (dS > 1e-12) {
                delta = (price_up_S - price_down_S) / (2.0 * dS);
                gamma = (price_up_S - 2.0 * price_mid + price_down_S) / (dS * dS);
            }

            // Vega
            sigma = orig_sigma + _dSigma;
            double price_up_sigma = this->_price_impl();
            double sigma_down = std::max(orig_sigma - _dSigma, 1e-9); // Ensure sigma > 0
            sigma = sigma_down;
            double price_down_sigma = this->_price_impl();
            sigma = orig_sigma; // Restore sigma
            double actual_dSigma = (orig_sigma + _dSigma) - sigma_down;
            if (actual_dSigma > 1e-12) {
                vega = (price_up_sigma - price_down_sigma) / actual_dSigma;
                vega /= 100.0; // per 1% change
            }

            // Theta
            T = std::max(orig_T - _dT_years, 1e-9); // Ensure T > 0
            double price_down_T = this->_price_impl();
            T = orig_T; // Restore T
            if (_dT_years > 1e-12) {
                theta = -(price_mid - price_down_T) / _dT_years / 365.25; // per calendar day
            }

            // Rho
            r = orig_r + _dr;
            double price_up_r = this->_price_impl();
            double r_down = orig_r - _dr; // Consider allowing negative rates if needed
            r = r_down;
            double price_down_r = this->_price_impl();
            r = orig_r; // Restore r
            if (_dr > 1e-12) {
                rho = (price_up_r - price_down_r) / (2.0 * _dr);
                rho /= 100.0; // per 1% change
            }
        } catch (...) {
            // If any _price_impl fails during Greek calc, leave Greeks as NAN
            // Restore parameters just in case
            S = orig_S; T = orig_T; sigma = orig_sigma; r = orig_r;
        }

        return {
            {"delta", delta}, {"gamma", gamma}, {"vega", vega},
            {"theta", theta}, {"rho", rho}
        };
    }

    std::map<std::string, double> calculate_all() {
        auto start = std::chrono::high_resolution_clock::now();
        double price = NAN;
        std::map<std::string, double> greeks;

        try {
            price = _price_impl();
            greeks = _greeks_impl();
        } catch (...) {
            // Handle exceptions during pricing or greek calculation if necessary
            // For now, results will contain NAN
            /* Handle */
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        std::map<std::string, double> results = {{"price", price}};
        results.insert(greeks.begin(), greeks.end()); // Add calculated greeks
        results["calc_time_sec"] = duration.count();

        return results;
    }
};


class CRRBinomialAmericanPricer : public AmericanOptionPricerBase {
public:
    CRRBinomialAmericanPricer(double s, double k, double t, double r_, double sigma_, double q_, OptionType type, int n_steps) :
        AmericanOptionPricerBase(s, k, t, r_, sigma_, q_, type)
    {
        if (n_steps < 1) throw std::invalid_argument("N must be >= 1.");
        N_steps_member = n_steps;
    }

    double _price_impl() override {
        int steps = N_steps_member;
        double dt = T / static_cast<double>(steps);
        double u = std::exp(sigma * std::sqrt(dt));
        double d = 1.0 / u;
        double exp_rqdt = std::exp((r - q) * dt);
        double p_den = u - d;
        double p = (std::abs(p_den) < 1e-15) ? 0.5 : (exp_rqdt - d) / p_den;

        // Ensure probability is within bounds
        p = std::max(1e-9, std::min(p, 1.0 - 1e-9));
        double one_minus_p = 1.0 - p;
        double discount = std::exp(-r * dt);

        std::vector<double> option_values(steps + 1);

        // Initialize values at expiration
        for (int j = 0; j <= steps; ++j) {
            double price_at_exp = S * std::pow(d, steps - j) * std::pow(u, j);
            option_values[j] = (option_type == OptionType::CALL)
                               ? std::max(0.0, price_at_exp - K)
                               : std::max(0.0, K - price_at_exp);
        }

        // Backward induction
        for (int i = steps - 1; i >= 0; --i) {
            for (int j = 0; j <= i; ++j) {
                double expected_value = discount * (p * option_values[j + 1] + one_minus_p * option_values[j]);
                double current_price = S * std::pow(d, i - j) * std::pow(u, j);
                double intrinsic_value = (option_type == OptionType::CALL)
                                         ? std::max(0.0, current_price - K)
                                         : std::max(0.0, K - current_price);
                option_values[j] = std::max(expected_value, intrinsic_value); // Check for early exercise
            }
        }
        return option_values[0];
    }
};


class LeisenReimerAmericanPricer : public AmericanOptionPricerBase {
public:
    LeisenReimerAmericanPricer(double s, double k, double t, double r_, double sigma_, double q_, OptionType type, int n_steps) :
        AmericanOptionPricerBase(s, k, t, r_, sigma_, q_, type)
    {
        N_steps_member = (n_steps % 2 == 0) ? n_steps + 1 : n_steps; // Ensure N is odd
        if (N_steps_member < 1) throw std::invalid_argument("N must be >= 1.");
    }

private:
    // Peizer-Pratt inversion method 2
    double _peizer_pratt_inversion(double z, int n) {
        double n_float = static_cast<double>(n);
        double epsilon = 1e-12; // Small number for stability checks

        if (n_float < 1) return 0.5; // Or handle as error

        double den_term_safe = n_float + 1.0/3.0 + 0.1 / (n_float + 1.0) + epsilon;
        if (std::abs(den_term_safe) < epsilon) return 0.5; // Avoid division by zero

        double term1_sq = std::pow(z / den_term_safe, 2.0);
        double term2 = n_float + 1.0/6.0;

        // Clamp exponent argument to prevent overflow/underflow in exp
        double exp_arg = std::max(-700.0, std::min(-term1_sq * term2, 700.0));

        double inner_sqrt = std::max(1.0 - std::exp(exp_arg), 0.0); // Ensure non-negative under sqrt

        double sign_z = (z > 0) ? 1.0 : ((z < 0) ? -1.0 : 0.0);
        return 0.5 + sign_z * 0.5 * std::sqrt(inner_sqrt);
    }

public:
    double _price_impl() override {
        int steps = N_steps_member;
        double dt = T / static_cast<double>(steps);
        double discount = std::exp(-r * dt);

        double vol_sqrt_T = sigma * std::sqrt(T);
        if (vol_sqrt_T < 1e-9) vol_sqrt_T = 1e-9; // Avoid division by zero

        double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_T;
        double d2 = d1 - vol_sqrt_T;

        // Calculate inverted probabilities using Peizer-Pratt
        double p_d1_inv = _peizer_pratt_inversion(d1, steps);
        double p_d2_inv = _peizer_pratt_inversion(d2, steps);

        // Clip probabilities to avoid issues near 0 or 1
        double epsilon = 1e-9;
        p_d1_inv = std::max(epsilon, std::min(p_d1_inv, 1.0 - epsilon));
        p_d2_inv = std::max(epsilon, std::min(p_d2_inv, 1.0 - epsilon));

        // Calculate LR tree parameters u and d
        double drift_adj = std::exp((r - q) * dt);
        double u = (p_d2_inv > epsilon) ? drift_adj * (p_d1_inv / p_d2_inv) : drift_adj ; // Avoid division by zero
        double p_d2_inv_minus_1 = 1.0 - p_d2_inv;
        double d;
        if (std::abs(p_d2_inv_minus_1) < epsilon) {
             d = (std::abs(u) > epsilon) ? drift_adj / u : drift_adj; // Avoid division by zero
        } else {
            d = (drift_adj - p_d2_inv * u) / p_d2_inv_minus_1;
        }

        double p_exp = p_d2_inv; // Risk-neutral probability in LR tree
        double one_minus_p_exp = 1.0 - p_exp;

        std::vector<double> option_values(steps + 1);

        // Initialize values at expiration
        for (int j = 0; j <= steps; ++j) {
            double price_at_exp = S * std::pow(d, steps - j) * std::pow(u, j);
            option_values[j] = (option_type == OptionType::CALL)
                               ? std::max(0.0, price_at_exp - K)
                               : std::max(0.0, K - price_at_exp);
        }

        // Backward induction (same as CRR, but with LR parameters)
        for (int i = steps - 1; i >= 0; --i) {
            for (int j = 0; j <= i; ++j) {
                double expected_value = discount * (p_exp * option_values[j + 1] + one_minus_p_exp * option_values[j]);
                double current_price = S * std::pow(d, i - j) * std::pow(u, j);
                double intrinsic_value = (option_type == OptionType::CALL)
                                         ? std::max(0.0, current_price - K)
                                         : std::max(0.0, K - current_price);
                option_values[j] = std::max(expected_value, intrinsic_value); // Check for early exercise
            }
        }
        return option_values[0];
    }
};


class BjerksundStensland2002Pricer : public AmericanOptionPricerBase {
public:
    BjerksundStensland2002Pricer(double s, double k, double t, double r_, double sigma_, double q_, OptionType type) :
        AmericanOptionPricerBase(s, k, t, r_, sigma_, q_, type)
    {}

private:
    // Helper for standard European BS price
    double _bs_price(double S_in, double K_in, double T_in, double r_in, double q_in, double sigma_in, OptionType opt_type) {
        double tol = 1e-9;
        if (T_in < tol) {
            return std::max(0.0, S_in - K_in) * (opt_type == OptionType::CALL ? 1.0 : 0.0) +
                   std::max(0.0, K_in - S_in) * (opt_type == OptionType::PUT ? 1.0 : 0.0);
        }
        sigma_in = std::max(sigma_in, tol); // Ensure sigma is positive
        double vol_sqrt_T = sigma_in * std::sqrt(T_in);
        if (vol_sqrt_T < tol) { // Avoid division by zero if T or sigma is extremely small
             return std::max(0.0, S_in - K_in) * (opt_type == OptionType::CALL ? 1.0 : 0.0) +
                   std::max(0.0, K_in - S_in) * (opt_type == OptionType::PUT ? 1.0 : 0.0);
        }

        double d1 = (std::log(S_in / K_in) + (r_in - q_in + 0.5 * sigma_in * sigma_in) * T_in) / vol_sqrt_T;
        double d2 = d1 - vol_sqrt_T;

        if (opt_type == OptionType::CALL) {
            return S_in * std::exp(-q_in * T_in) * normalCDF(d1) - K_in * std::exp(-r_in * T_in) * normalCDF(d2);
        } else { // PUT
            return K_in * std::exp(-r_in * T_in) * normalCDF(-d2) - S_in * std::exp(-q_in * T_in) * normalCDF(-d1);
        }
    }

    // Phi function as used in BS 2002 (following Haug formulation generally)
    double _phi(double S_phi, double T_phi, double gamma, double H, double I, double r_phi, double q_phi, double sigma_phi) {
        T_phi = std::max(T_phi, 1e-9); // Ensure T > 0
        sigma_phi = std::max(sigma_phi, 1e-9); // Ensure sigma > 0
        double vol_sqrt_T = sigma_phi * std::sqrt(T_phi);

        double lambda_ = (-r_phi + gamma * q_phi + 0.5 * gamma * (gamma - 1.0) * sigma_phi * sigma_phi) * T_phi;
        double d1_num = std::log(S_phi / H) + (q_phi + (gamma - 0.5) * sigma_phi * sigma_phi) * T_phi;
        double d1 = d1_num / vol_sqrt_T;

        if (I <= 0 || S_phi <= 0) return 0.0; // Avoid log(non-positive)

        double d1_shifted = d1 - (2.0 * std::log(I / S_phi)) / vol_sqrt_T;
        double kappa_pow = 2.0 * (r_phi - q_phi) / (sigma_phi * sigma_phi) + 2.0 * gamma - 1.0;

        lambda_ = std::max(-700.0, std::min(lambda_, 700.0)); // Clamp exponent for exp

        double term1 = std::exp(lambda_) * std::pow(S_phi, gamma);
        double term2 = normalCDF(d1);

        double I_S_ratio = I / S_phi;
        double term3_mult = 0.0;
        if (I_S_ratio > 0) {
            double base_clamp = std::max(I_S_ratio, 1e-10); // Avoid issues if ratio is extremely small
            try {
                term3_mult = std::pow(base_clamp, kappa_pow);
            } catch (const std::overflow_error&) {
                term3_mult = std::numeric_limits<double>::infinity(); // Handle potential overflow
            }
            if (std::isnan(term3_mult)) term3_mult = 0.0; // Handle potential NaN from pow
        }

        double n_d1_shifted = normalCDF(d1_shifted);
        double term3 = term3_mult * n_d1_shifted;

        // Handle potential inf * 0 = NaN cases
        if (std::isinf(term3_mult) && std::abs(n_d1_shifted) < 1e-12) term3 = 0.0;
        else if (std::isinf(term3) || std::isnan(term3)) term3 = 0.0;

        return term1 * (term2 - term3);
    }

public:
    double _price_impl() override {
        double tol = 1e-9;
        if (T < tol) { // Handle expiry
             return std::max(0.0, S - K) * (option_type == OptionType::CALL ? 1.0 : 0.0) +
                   std::max(0.0, K - S) * (option_type == OptionType::PUT ? 1.0 : 0.0);
        }

        // Use local copies for clarity within the implementation
        double S_ = S, K_ = K, T_ = T, r_ = r, q_ = q, sigma_ = sigma;
        double sigma_sq = sigma_ * sigma_;

        if (option_type == OptionType::CALL) {
            // Calculate beta parameter
             double beta_num_sqrt_term = std::pow((r_ - q_) / sigma_sq - 0.5, 2.0) + 2.0 * r_ / sigma_sq;
             if (beta_num_sqrt_term < 0) beta_num_sqrt_term = 0; // Ensure non-negative
             double beta_num_sqrt = std::sqrt(beta_num_sqrt_term);
             double beta = (0.5 - (r_ - q_) / sigma_sq) + beta_num_sqrt;

            // Calculate B0 and Binf (critical prices related to dividends/rates)
            double B0 = (q_ <= 0) ? K_ : std::max(K_, (r_ / q_) * K_); // Handle q=0 case
            double Binf = (std::abs(beta - 1.0) > tol) ? (beta / (beta - 1.0)) * K_ : std::numeric_limits<double>::infinity(); // Handle beta=1 case

            // Calculate trigger price I
             double h_T_num_sqrt = std::sqrt(T_);
             double h_T_den = Binf - B0;
             if (std::abs(h_T_den) < tol || std::isinf(Binf)) { // If Binf = B0 or Binf is infinite, use European price
                 return _bs_price(S_, K_, T_, r_, q_, sigma_, OptionType::CALL);
             }
             // Exponent in Haug's h(T) formulation, using K
             double h_T_exponent = -((r_ - q_) * T_ + 2.0 * sigma_ * h_T_num_sqrt * K_ / h_T_den);
             h_T_exponent = std::max(-700.0, std::min(h_T_exponent, 700.0)); // Clamp exponent
             double I = B0 + (Binf - B0) * (1.0 - std::exp(h_T_exponent));

            // Price calculation based on trigger price
            if (S_ >= I) {
                return std::max(0.0, S_ - K_); // Should exercise immediately
            } else {
                double eur_price = _bs_price(S_, K_, T_, r_, q_, sigma_, OptionType::CALL);
                double phi_val = _phi(S_, T_, beta, I, I, r_, q_, sigma_); // Note: H=I for Call
                // Return max of approximated American price, European price, and intrinsic value
                return std::max({eur_price + phi_val, eur_price, S_ - K_});
            }
        } else { // PUT Option
            // Calculate gamma_put parameter (note the sign difference from call's beta)
             double gamma_put_num_sqrt_term = std::pow((r_ - q_) / sigma_sq - 0.5, 2.0) + 2.0 * r_ / sigma_sq;
             if (gamma_put_num_sqrt_term < 0) gamma_put_num_sqrt_term = 0;
             double gamma_put_num_sqrt = std::sqrt(gamma_put_num_sqrt_term);
             double gamma_put = (0.5 - (r_ - q_) / sigma_sq) - gamma_put_num_sqrt;

            // Calculate B0_put and Binf_put (critical prices)
            double B0_put = (q_ <= 0) ? K_ : std::min(K_, (r_ / q_) * K_); // Handle q=0
            double Binf_put = 0.0; // Binf for puts is 0

            // Calculate trigger price I_put
             double h_T_num_sqrt = std::sqrt(T_);
             double h_T_den = B0_put - Binf_put; // = B0_put since Binf_put = 0
             if (std::abs(h_T_den) < tol) { // Should only happen if K=0 or r=0 and q>0? Return BS price.
                  return _bs_price(S_, K_, T_, r_, q_, sigma_, OptionType::PUT);
             }
             // Exponent in Haug's h(T) formulation for puts, using K
             double h_T_exponent_put = ((r_ - q_) * T_ - 2.0 * sigma_ * h_T_num_sqrt * K_ / h_T_den);
             h_T_exponent_put = std::max(-700.0, std::min(h_T_exponent_put, 700.0)); // Clamp exponent
             double I_put = Binf_put + (B0_put - Binf_put) * (1.0 - std::exp(h_T_exponent_put));

            // Price calculation based on trigger price
            if (S_ <= I_put) {
                return std::max(0.0, K_ - S_); // Should exercise immediately
            } else {
                double eur_price = _bs_price(S_, K_, T_, r_, q_, sigma_, OptionType::PUT);
                double phi_val_put = _phi(S_, T_, gamma_put, I_put, I_put, r_, q_, sigma_); // Note: H=I for Put
                // Return max of approximated American price, European price, and intrinsic value
                return std::max({eur_price + phi_val_put, eur_price, K_ - S_});
            }
        }
    }
};

// --- Main Benchmark Execution ---
int main() {
    std::cout << "--- Starting C++ Option Model Benchmark (Reads Discrete Div Input) ---" << std::endl;
    std::cout << " Input Dir (Hardcoded): " << INPUT_DIR << std::endl;
    std::cout << " Output Dir (Hardcoded): " << OUTPUT_DIR << std::endl;
    std::cout << " Runs per model: " << BENCHMARK_N_RUNS << std::endl;
    std::cout << " Binomial steps: " << BINOMIAL_N_STEPS << std::endl;

    // --- Load Benchmark Input Data ---
    std::vector<std::map<std::string, std::string>> benchmark_inputs;
    std::vector<std::string> headers;
    try {
        // *** READ THE CORRECT INPUT FILE ***
        std::string input_filename = INPUT_DIR + "/benchmark_input_data_discrete_div.csv";
        std::ifstream input_file(input_filename);
        if (!input_file.is_open()) throw std::runtime_error("Cannot open benchmark input file: " + input_filename);
        std::string line, word;
        if (std::getline(input_file, line)) {
             if (!line.empty() && line.back() == '\r') line.pop_back();
             std::stringstream ss_header(line); while (std::getline(ss_header, word, ',')) { headers.push_back(word); } }
        else { throw std::runtime_error("Input file empty/header missing."); }
        while (std::getline(input_file, line)) {
             if (!line.empty() && line.back() == '\r') { line.pop_back(); } if (line.empty()) continue;
             std::stringstream ss(line); std::map<std::string, std::string> row_map;
             for (size_t i = 0; i < headers.size(); ++i) { if (!std::getline(ss, word, ',')) { word = ""; } word.erase(0, word.find_first_not_of(" \t\n\r\f\v")); word.erase(word.find_last_not_of(" \t\n\r\f\v") + 1); row_map[headers[i]] = word; }
             if (!row_map.empty()) { benchmark_inputs.push_back(row_map); }
        } input_file.close();
        std::cout << "Loaded " << benchmark_inputs.size() << " benchmark input rows from " << input_filename << "." << std::endl;
    } catch (const std::exception& e) { std::cerr << "FATAL: Error loading benchmark data: " << e.what() << std::endl; return 1; }
    if (benchmark_inputs.empty()) { std::cerr << "FATAL: No benchmark input data." << std::endl; return 1; }

    // --- Define Models ---
    std::map<std::string, std::function<std::unique_ptr<AmericanOptionPricerBase>(double, double, double, double, double, double, OptionType)>> model_factory;
    model_factory["CRR"] = [](double s, double k, double t, double r, double sigma, double q, OptionType type) { return std::make_unique<CRRBinomialAmericanPricer>(s, k, t, r, sigma, q, type, BINOMIAL_N_STEPS); };
    model_factory["LeisenReimer"] = [](double s, double k, double t, double r, double sigma, double q, OptionType type) { return std::make_unique<LeisenReimerAmericanPricer>(s, k, t, r, sigma, q, type, BINOMIAL_N_STEPS); };
    model_factory["BS2002"] = [](double s, double k, double t, double r, double sigma, double q, OptionType type) { return std::make_unique<BjerksundStensland2002Pricer>(s, k, t, r, sigma, q, type); };

    std::vector<BenchmarkResult> benchmark_results; int processed_rows = 0;

    // --- Iterate Through Input Rows ---
    for (size_t i = 0; i < benchmark_inputs.size(); ++i) {
        const auto& input_row_map = benchmark_inputs[i];
        std::cout << "\rProcessing Input Row " << (i + 1) << "/" << benchmark_inputs.size() << "..." << std::flush;
        try {
            // --- Extract parameters directly from the map ---
            // Note: S is the dividend-adjusted S, q should be 0
            double spot_price = std::stod(input_row_map.at("S"));
            double strike_price = std::stod(input_row_map.at("K"));
            double time_to_expiry = std::stod(input_row_map.at("T"));
            double risk_free_rate = std::stod(input_row_map.at("r"));
            double hist_vol = std::stod(input_row_map.at("sigma"));
            double div_yield = std::stod(input_row_map.at("q")); // Should be 0
            std::string option_type_str = input_row_map.at("option_type");
            OptionType opt_type_enum = (option_type_str == "call") ? OptionType::CALL : OptionType::PUT;

            // Validate inputs
            if (std::isnan(spot_price) || std::isnan(strike_price) || std::isnan(time_to_expiry) ||
                std::isnan(risk_free_rate) || std::isnan(hist_vol) || std::isnan(div_yield) ||
                time_to_expiry <= 0 || hist_vol <= 1e-6 || spot_price <= 0 || strike_price <=0 ) {
                 std::cerr << "\n Skipping row " << i + 1 << " invalid numeric inputs."; continue;
            }
            // Sanity check q (optional)
            if (std::abs(div_yield) > 1e-9) {
                std::cerr << "\n Warning: Row " << i + 1 << ": Input q is " << div_yield << " but expected 0 for discrete div adj.";
            }


            // --- Run Models ---
            for (const auto& pair : model_factory) {
                const std::string& model_name = pair.first; const auto& factory_func = pair.second;
                std::vector<std::map<std::string, double>> run_results_list; double total_time = 0;
                for (int run = 0; run < BENCHMARK_N_RUNS; ++run) {
                    try {
                         // Use the loaded parameters directly
                         auto model_instance = factory_func(spot_price, strike_price, time_to_expiry, risk_free_rate, hist_vol, div_yield, opt_type_enum);
                         std::map<std::string, double> result_dict = model_instance->calculate_all(); run_results_list.push_back(result_dict);
                         if (result_dict.count("calc_time_sec")) { total_time += result_dict.at("calc_time_sec"); } else { run_results_list.back()["calc_time_sec"] = NAN; }
                    } catch (const std::exception& model_err) { std::cerr << "\n   ERROR running " << model_name << ": " << model_err.what(); std::map<std::string, double> error_result; error_result["calc_time_sec"] = NAN; run_results_list.push_back(error_result); break; } }

                // Store results
                for (int run = 0; run < run_results_list.size(); ++run) {
                    const auto& result_dict = run_results_list[run]; BenchmarkResult res;
                    // Populate BenchmarkResult struct from input_row_map and result_dict
                    res.underlying_ticker = input_row_map.at("underlying_ticker"); res.option_ticker = input_row_map.at("option_ticker"); res.target_dte_group = std::stoi(input_row_map.at("target_dte_group")); res.option_type = option_type_str; res.timestamp = input_row_map.at("timestamp");
                    res.model = model_name; res.binomial_steps = (model_name == "CRR" || model_name == "LeisenReimer") ? BINOMIAL_N_STEPS : 0; res.run_number = run + 1;
                    res.calc_time_sec = result_dict.count("calc_time_sec") ? result_dict.at("calc_time_sec") : NAN;
                    res.calculated_price = result_dict.count("price") ? result_dict.at("price") : NAN; res.delta = result_dict.count("delta") ? result_dict.at("delta") : NAN; res.gamma = result_dict.count("gamma") ? result_dict.at("gamma") : NAN; res.vega = result_dict.count("vega") ? result_dict.at("vega") : NAN; res.theta = result_dict.count("theta") ? result_dict.at("theta") : NAN; res.rho = result_dict.count("rho") ? result_dict.at("rho") : NAN;
                    res.input_S = spot_price; res.input_K = strike_price; res.input_T = time_to_expiry; res.input_r = risk_free_rate; res.input_sigma = hist_vol; res.input_q = div_yield;
                    benchmark_results.push_back(res); }
            } // End model loop
            processed_rows++;
        } catch (const std::out_of_range& oor) { std::cerr << "\n Skip row " << i + 1 << " missing key: " << oor.what(); } catch (const std::invalid_argument& ia) { std::cerr << "\n Skip row " << i + 1 << " conv error: " << ia.what(); } catch (const std::exception& e) { std::cerr << "\n Skip row " << i + 1 << " error: " << e.what(); }
    } // End input row loop

    std::cout << "\nBenchmarking finished. Processed " << processed_rows << " input rows." << std::endl;

    // --- Save Results ---
    if (!benchmark_results.empty()) {
        std::string output_filename = OUTPUT_DIR + "/cpp_runtime_benchmark_results.csv";
        std::ofstream out_file(output_filename); if (!out_file.is_open()) { std::cerr << "FATAL: Cannot open output file: " << output_filename << std::endl; return 1; }
        out_file << "underlying_ticker,option_ticker,target_dte_group,option_type,timestamp,model,binomial_steps,run_number,calc_time_sec,calculated_price,delta,gamma,vega,theta,rho,input_S,input_K,input_T,input_r,input_sigma,input_q\n";
        out_file << std::fixed << std::setprecision(9); // High precision for output
        for (const auto& res : benchmark_results) {
            out_file << res.underlying_ticker << "," << res.option_ticker << "," << res.target_dte_group << "," << res.option_type << "," << res.timestamp << ","
                     << res.model << "," << res.binomial_steps << "," << res.run_number << "," << res.calc_time_sec << ","
                     << res.calculated_price << "," << res.delta << "," << res.gamma << "," << res.vega << "," << res.theta << "," << res.rho << ","
                     << res.input_S << "," << res.input_K << "," << res.input_T << "," << res.input_r << "," << res.input_sigma << "," << res.input_q << "\n";
        } out_file.close();
        std::cout << "Saved C++ benchmark results (" << benchmark_results.size() << " rows) to: " << output_filename << std::endl;
    } else { std::cout << "No C++ benchmark results generated." << std::endl; }
    return 0;
}