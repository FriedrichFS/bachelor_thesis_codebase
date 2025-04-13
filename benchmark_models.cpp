-) || std::isnan(time_to_expiry) ||
                std::isnan(risk_free_rate) || std::isnan(hist_vol) || std::isnan(div_yield) ||
                time_to_expiry <= 0 || hist_vol <= 1e-6 || spot_price <= 0 || strike_price <=0 ) {
                 std::cerr << "\n Skipping row " << i + 1 << " invalid numeric inputs."; continue;
            }
            // ? Keep? Sanity check q 
            if (std::abs(div_yield) > 1e-9) {
                std::cerr << "\n Warning: Row " << i + 1 << ": Input q is " << div_yield << " but expected 0 for discrete div adj.";
            }


            // !--- Run Models ---
            for (const auto& pair : model_factory) {
                const std::string& model_name = pair.first; const auto& factory_func = pair.second;
                std::vector<std::map<std::string, double>> run_results_list; double total_time = 0;
                for (int run = 0; run < BENCHMARK_N_RUNS; ++run) {
                    try {
                         // Uses the loaded parameters directly
                         auto model_instance = factory_func(spot_price, strike_price, time_to_expiry, risk_free_rate, hist_vol, div_yield, opt_type_enum);
                         std::map<std::string, double> result_dict = model_instance->calculate_all(); run_results_list.push_back(result_dict);
                         if (result_dict.count("calc_time_sec")) { total_time += result_dict.at("calc_time_sec"); } else { run_results_list.back()["calc_time_sec"] = NAN; }
                    } catch (const std::exception& model_err) { std::cerr << "\n   ERROR running " << model_name << ": " << model_err.what(); std::map<std::string, double> error_result; error_result["calc_time_sec"] = NAN; run_results_list.push_back(error_result); break; } }

                // Stores results
                for (int run = 0; run < run_results_list.size(); ++run) {
                    const auto& result_dict = run_results_list[run]; BenchmarkResult res;
                    // Populates BenchmarkResult struct from input_row_map and result_dict
                    res.underlying_ticker = input_row_map.at("underlying_ticker"); res.option_ticker = input_row_map.at("option_ticker"); res.target_dte_group = std::stoi(input_row_map.at("target_dte_group")); res.option_type = option_type_str; res.timestamp = input_row_map.at("timestamp");
                    res.model = model_name; res.binomial_steps = (model_name == "CRR" || model_name == "LeisenReimer") ? BINOMIAL_N_STEPS : 0; res.run_number = run + 1;
                    res.calc_time_sec = result_dict.count("calc_time_sec") ? result_dict.at("calc_time_sec") : NAN;
                    res.calculated_price = result_dict.count("price") ? result_dict.at("price") : NAN; res.delta = result_dict.count("delta") ? result_dict.at("delta") : NAN; res.gamma = result_dict.count("gamma") ? result_dict.at("gamma") : NAN; res.vega = result_dict.count("vega") ? result_dict.at("vega") : NAN; res.theta = result_dict.count("theta") ? result_dict.at("theta") : NAN; res.rho = result_dict.count("rho") ? result_dict.at("rho") : NAN;
                    res.input_S = spot_price; res.input_K = strike_price; res.input_T = time_to_expiry; res.input_r = risk_free_rate; res.input_sigma = hist_vol; res.input_q = div_yield;
                    benchmark_results.push_back(res); }
            } // ! End model loop here
            processed_rows++;
        } catch (const std::out_of_range& oor) { std::cerr << "\n Skip row " << i + 1 << " missing key: " << oor.what(); } catch (const std::invalid_argument& ia) { std::cerr << "\n Skip row " << i + 1 << " conv error: " << ia.what(); } catch (const std::exception& e) { std::cerr << "\n Skip row " << i + 1 << " error: " << e.what(); }
    } // Big input loop :: End!

    std::cout << "\nBenchmarking finished. Processed " << processed_rows << " input rows." << std::endl;

    // !--- Save Results ---
    if (!benchmark_results.empty()) {
        std::string output_filename = OUTPUT_DIR + "/cpp_runtime_benchmark_results.csv";
        std::ofstream out_file(output_filename); if (!out_file.is_open()) { std::cerr << "FATAL: Cannot open output file: " << output_filename << std::endl; return 1; }
        out_file << "underlying_ticker,option_ticker,target_dte_group,option_type,timestamp,model,binomial_steps,run_number,calc_time_sec,calculated_price,delta,gamma,vega,theta,rho,input_S,input_K,input_T,input_r,input_sigma,input_q\n";
        out_file << std::fixed << std::setprecision(9); // High precision for output, Important for benchmark reuslts eval
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