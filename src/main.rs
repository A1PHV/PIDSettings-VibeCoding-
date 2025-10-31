mod log_parser;
mod pid_calculator;
mod models;

use clap::{Parser, Subcommand};
use anyhow::Result;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "pid_tuner")]
#[command(about = "Ardupilot PID coefficient calculator from flight logs", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze log file and calculate optimal PID coefficients
    Analyze {
        /// Path to Ardupilot log file (.log or .csv format)
        #[arg(short, long)]
        input: PathBuf,

        /// Output file for calculated PID coefficients
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Axis to tune (roll, pitch, yaw, alt, all)
        #[arg(short, long, default_value = "all")]
        axis: String,

        /// Optimization method (ziegler-nichols, relay, manual)
        #[arg(short, long, default_value = "ziegler-nichols")]
        method: String,
    },

    /// Extract flight data from log to CSV
    Extract {
        /// Path to Ardupilot log file
        #[arg(short, long)]
        input: PathBuf,

        /// Output CSV file
        #[arg(short, long)]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Analyze { input, output, axis, method } => {
            println!("🚁 Analyzing Ardupilot log: {}", input.display());
            println!("📊 Axis: {}", axis);
            println!("🔧 Method: {}", method);

            // Parse log file
            let flight_data = log_parser::parse_log(&input)?;
            println!("✅ Parsed {} data points", flight_data.len());

            // Calculate PID coefficients
            let pid_results = pid_calculator::calculate_pid_coefficients(
                &flight_data,
                &axis,
                &method
            )?;

            // Print results
            println!("\n📈 Calculated PID Coefficients:");
            println!("{}", pid_results);

            // Save to file if specified
            if let Some(output_path) = output {
                pid_results.save_to_file(&output_path)?;
                println!("\n💾 Results saved to: {}", output_path.display());
            }

            Ok(())
        },

        Commands::Extract { input, output } => {
            println!("📝 Extracting data from: {}", input.display());

            let flight_data = log_parser::parse_log(&input)?;
            log_parser::export_to_csv(&flight_data, &output)?;

            println!("✅ Data exported to: {}", output.display());
            Ok(())
        }
    }
}
