mod runtime;
#[allow(dead_code)]
mod contract;
#[allow(dead_code)]
mod model;
mod sampling;
mod bridge;
#[allow(dead_code)]
mod world;
mod physics;
mod observe;
mod checks;
mod experience;
mod learner;
mod ggd_ppo;
mod ggd_arenas;
mod cell;
mod gradients;
mod artifacts;
mod trace;
mod fidelity;
mod aggro_fidelity;
mod requiem_fidelity;
mod navigation;
mod timing_observe;
mod enemy_observe;
mod selfplay;
mod evaluation;
mod departure;
mod policy_trace;
mod lane_fidelity;
mod lr_diagnostic;
mod startup_audit;
mod startup_step;
mod critic_dataset;
mod scripted_teacher;
mod teacher_combat;
mod teacher_v10;
mod teacher_eval;
mod combat_observe;
mod imitation;
mod demo_telemetry;
mod ggd_metrics;
mod ggd_diagnostic;
#[cfg(test)]
mod gpu_tests;

use clap::{Parser,Subcommand};
use std::path::PathBuf;
use burn::module::Module;

#[derive(Parser)]
#[command(version,about="GPU SF Duel simulator and recurrent PPO")]
struct Cli {#[command(subcommand)] command:Command}

#[cfg(test)]
mod cli_tests {
    use super::*;
    #[test]
    fn ggd_selection_preserves_default_and_rejects_conflicting_flags(){
        for (flags,greedy,zero) in [(vec![],true,false),(vec!["--argmax"],true,false),(vec!["--sampling","--zero-previous"],false,true)]{
            let mut args=vec!["sf-duel-gpu","evaluate-ggd","engine","weights","output"];args.extend(flags);
            let Command::EvaluateGgd{argmax,sampling,zero_previous,..}=Cli::try_parse_from(args).unwrap().command else{panic!("wrong command")};
            assert_eq!(argmax||!sampling,greedy);assert_eq!(zero_previous,zero);
        }
        assert!(Cli::try_parse_from(["sf-duel-gpu","evaluate-ggd","engine","weights","output","--argmax","--sampling"]).is_err());
    }
}

#[derive(Subcommand)]
enum Command {
    CollectTeacher {config:PathBuf,output:PathBuf},
    CollectDagger {config:PathBuf,output:PathBuf},
    TrainGgd {config:PathBuf,output:PathBuf},
    TrainIl {config:PathBuf,output:PathBuf,#[arg(long)]max_updates:Option<usize>},
    EvaluateGgd {engine:PathBuf,weights:PathBuf,output:PathBuf,#[arg(long,default_value_t=8)]matches:usize,#[arg(long,default_value_t=600.)]seconds:f32,#[arg(long,default_value="teacher")]mode:String,#[arg(long,default_value_t=20261103)]seed:u32,#[arg(long)]trace:bool,#[arg(long)]record_teacher:bool,
        #[arg(long,conflicts_with="sampling",help="Greedy selection (also the legacy default)")]argmax:bool,#[arg(long,help="Sample all conditional action heads")]sampling:bool,#[arg(long)]zero_previous:bool,#[arg(long,value_enum,default_value="v9")]teacher_version:teacher_v10::Version},
    EvaluateTeachers {engine:PathBuf,output:PathBuf,#[arg(long,default_value_t=200)]matches:usize,#[arg(long,default_value_t=600.)]seconds:f32,#[arg(long,default_value_t=20261103)]seed:u32,#[arg(long,value_enum,default_value="v9")]left:teacher_v10::Version,#[arg(long,value_enum,default_value="v9")]right:teacher_v10::Version,#[arg(long)]swap:bool,#[arg(long)]trace:bool},
    TeacherCheck {engine:PathBuf,output:PathBuf},
    Teacher {engine:PathBuf,output:PathBuf,#[arg(long,default_value_t=100)]matches:usize,#[arg(long,default_value_t=600.)]seconds:f32,#[arg(long,default_value_t=20261102)]seed:u32,#[arg(long,default_value_t=900.)]wall_seconds:f64,#[arg(long)]trace:bool},
    Qualify,
    Validate {bundle:PathBuf},
    #[command(hide=true)] ValidatePrevious {bundle:PathBuf,output:PathBuf},
    #[command(hide=true)] CheckGgdParity {weights:PathBuf,fixture:PathBuf,output:PathBuf},
    ValidateCombat {engine:PathBuf,reference:PathBuf},
    ValidateMovement {engine:PathBuf,reference:PathBuf},
    ReplayLane {engine:PathBuf,reference:PathBuf,output:PathBuf},
    ValidateAggro {engine:PathBuf,reference:PathBuf},
    ValidateRequiem {engine:PathBuf,reference:PathBuf,#[arg(long,default_value_t=0.)]fear_distance:f32,#[arg(long)]fear_from_caster:bool,#[arg(long)]fear_fixed_goal:bool},
    BuildNavigation {engine:PathBuf},
    CheckTiming {engine:PathBuf,fixture:PathBuf,output:PathBuf},
    CheckPublicSpells {engine:PathBuf,fixture:PathBuf,output:PathBuf},
    Import {bundle:PathBuf,output:PathBuf},
    Export {checkpoint:PathBuf,output:PathBuf},
    Replay {#[arg(required=true)]logs:Vec<PathBuf>,#[arg(long)]output:PathBuf,#[arg(long,default_value_t=0)]r#match:usize},
    Train {config:PathBuf,run:PathBuf,#[arg(long,default_value_t=0)]cycles:usize,#[arg(long)]resume:bool,#[arg(long)]draft:bool,#[arg(long,hide=true)]diagnostic_terms:Option<String>,#[arg(long,hide=true)]audit_rollouts:bool},
    Evaluate {config:PathBuf,output:PathBuf,#[arg(long)]no_traces:bool,#[arg(long)]argmax:bool,#[arg(long,allow_hyphen_values=true)]reset_origin:Option<f32>},
    CheckDeparture {config:PathBuf,output:PathBuf,#[arg(long)]argmax:bool,#[arg(long)]capture_start:bool,#[arg(long,allow_hyphen_values=true)]reset_origin:Option<f32>,#[arg(long,default_value_t=10.)]post_mid_seconds:f32,#[arg(long)]force_first_forward_move:bool},
    CompareLr {config:PathBuf,output:PathBuf},
    #[command(hide=true)] DiagnosePpo {config:PathBuf,output:PathBuf},
    #[command(hide=true)] AuditStartup {config:PathBuf,output:PathBuf},
    #[command(hide=true)] AuditStartupStep {audit:PathBuf,output:PathBuf},
    #[command(hide=true)] CollectCritic {config:PathBuf,output:PathBuf},
    Benchmark {config:PathBuf,output:PathBuf,#[arg(long,default_value_t=4096)]steps:usize,#[arg(long,default_value_t=128)]warmup:usize,
        #[arg(long)]simulation_only:bool,#[arg(long)]profile:bool,#[arg(long)]train:bool,#[arg(long,default_value_t=8)]cycles:usize,#[arg(long)]no_artifacts:bool,
        #[arg(long)]continuous_memory:bool,#[arg(long)]trace:bool},
    Stop {run:PathBuf},
    #[command(hide=true)] CellCheck,
    #[command(hide=true)] DumpInitialObservation {engine:PathBuf,output:PathBuf,#[arg(long,default_value_t=3)]ticks:usize},
    #[command(hide=true)] DumpIdleLane {engine:PathBuf,reference:PathBuf,output:PathBuf,#[arg(long,default_value_t=90)]seconds:usize},
    #[command(hide=true)] SimCheck {engine:PathBuf,#[arg(default_value_t=100)]matches:usize,#[arg(default_value_t=4)]steps:usize,#[arg(long)]inference:bool},
    #[command(hide=true)] TrainCheck {#[arg(default_value_t=100)]matches:usize,#[arg(default_value_t=1)]cycles:usize,#[arg(default_value_t=64)]batch:usize},
}
fn main()->anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).with_writer(std::io::stderr).init();
    let result=match Cli::parse().command {
        Command::CollectTeacher{config,output}=>imitation::collect(serde_json::from_slice(&std::fs::read(config)?)?,&output)?,
        Command::CollectDagger{config,output}=>imitation::collect_dagger(serde_json::from_slice(&std::fs::read(config)?)?,&output)?,
        Command::TrainGgd{config,output}=>ggd_ppo::run(serde_json::from_slice(&std::fs::read(config)?)?,&output)?,
        Command::TrainIl{config,output,max_updates}=>imitation::train(&config,&output,max_updates)?,
        Command::EvaluateGgd{engine,weights,output,matches,seconds,mode,seed,trace,record_teacher,argmax,sampling,zero_previous,teacher_version}=>imitation::evaluate(&engine,&weights,&output,matches,seconds,&mode,seed,trace,record_teacher,argmax||!sampling,zero_previous,teacher_version)?,
        Command::EvaluateTeachers{engine,output,matches,seconds,seed,left,right,swap,trace}=>teacher_eval::run(&engine,&output,matches,seconds,seed,left,right,swap,trace)?,
        Command::TeacherCheck{engine,output}=>scripted_teacher::check(&engine,&output)?,
        Command::Teacher{engine,output,matches,seconds,seed,wall_seconds,trace}=>scripted_teacher::run(&engine,&output,matches,seconds,seed,wall_seconds,trace)?,
        Command::Qualify=>runtime::qualification()?,
        Command::Validate{bundle}=>bridge::validate(&bundle,&runtime::device())?,
        Command::CheckGgdParity{weights,fixture,output}=>ggd_diagnostic::parity(&weights,&fixture,&output)?,
        Command::ValidatePrevious{bundle,output}=>{
            let report=bridge::validate_previous(&bundle,&runtime::device())?;
            artifacts::atomic_json(&output,&report)?;
            if report["passed"]!=true {anyhow::bail!("previous-action differential failed; see {}",output.display());}report
        },
        Command::ValidateCombat{engine,reference}=>{
            let report=fidelity::combat(&engine,&reference)?;
            if report["passed"]!=true {println!("{}",serde_json::to_string_pretty(&report)?);anyhow::bail!("Dota combat differential validation failed");}report
        },
        Command::ValidateMovement{engine,reference}=>{
            let report=fidelity::movement(&engine,&reference)?;
            if report["passed"]!=true {println!("{}",serde_json::to_string_pretty(&report)?);anyhow::bail!("Dota movement differential validation failed");}report
        },
        Command::ReplayLane{engine,reference,output}=>lane_fidelity::replay(&engine,&reference,&output)?,
        Command::ValidateAggro{engine,reference}=>{
            let report=aggro_fidelity::validate(&engine,&reference)?;
            if report["passed"]!=true {println!("{}",serde_json::to_string_pretty(&report)?);anyhow::bail!("Dota aggro differential validation failed");}report
        },
        Command::ValidateRequiem{engine,reference,fear_distance,fear_from_caster,fear_fixed_goal}=>{
            let report=requiem_fidelity::validate(&engine,&reference,fear_distance,fear_from_caster,fear_fixed_goal)?;
            if report["passed"]!=true {println!("{}",serde_json::to_string_pretty(&report)?);anyhow::bail!("Dota Requiem differential validation failed");}report
        },
        Command::BuildNavigation{engine}=>navigation::build(&engine)?,
        Command::CheckTiming{engine,fixture,output}=>timing_observe::check(&engine,&fixture,&output)?,
        Command::CheckPublicSpells{engine,fixture,output}=>enemy_observe::check(&engine,&fixture,&output)?,
        Command::Import{bundle,output}=>{
            let policy=bridge::import(&bundle,&runtime::device())?;
            if let Some(parent)=output.parent(){std::fs::create_dir_all(parent)?;}
            policy.into_record().save(&output)?;serde_json::json!({"model":output})
        }
        Command::Export{checkpoint,output}=>{
            let model_file=if checkpoint.is_dir(){checkpoint.join("model.bpk")}else{checkpoint};
            let policy=model::Policy::new(&runtime::device()).try_load_record(burn::store::ModuleRecord::load(model_file)?)?;
            let bundle=output.with_extension("tensors");bridge::export(&policy,&bundle)?;
            let root=PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
            let python=if cfg!(windows){"python"}else{"/home/zelt/dota-compat-check/bin/python"};
            let status=std::process::Command::new(python).arg(root.join("gpu/scripts/export_policy.py")).arg(&bundle).arg(&output).status()?;
            anyhow::ensure!(status.success(),"PyTorch export failed");serde_json::json!({"checkpoint":output})
        }
        Command::Train{config,run,cycles,resume,draft,diagnostic_terms,audit_rollouts}=>{
            let cfg=serde_json::from_slice(&std::fs::read(config)?)?;
            if audit_rollouts {anyhow::ensure!(diagnostic_terms.is_none(),"use ordinary PPO for audited counterexample");learner::run_audited(cfg,cycles,&run,resume,draft)?}else if let Some(terms)=diagnostic_terms {learner::run_loss_diagnostic(cfg,cycles,&run,resume,draft,&terms)?}else{learner::run(cfg,cycles,Some(&run),resume,draft)?}
        },
        Command::Evaluate{config,output,no_traces,reset_origin,argmax}=>evaluation::run(serde_json::from_slice(&std::fs::read(config)?)?,&output,!no_traces,reset_origin,argmax)?,
        Command::CheckDeparture{config,output,argmax,capture_start,reset_origin,post_mid_seconds,force_first_forward_move}=>departure::run(serde_json::from_slice(&std::fs::read(config)?)?,&output,argmax,capture_start,reset_origin,post_mid_seconds,force_first_forward_move)?,
        Command::CompareLr{config,output}=>lr_diagnostic::run(serde_json::from_slice(&std::fs::read(config)?)?,&output)?,
        Command::DiagnosePpo{config,output}=>lr_diagnostic::diagnose_updates(serde_json::from_slice(&std::fs::read(config)?)?,&output)?,
        Command::AuditStartup{config,output}=>startup_audit::run(serde_json::from_slice(&std::fs::read(config)?)?,&output)?,
        Command::AuditStartupStep{audit,output}=>startup_step::run(&audit,&output)?,
        Command::CollectCritic{config,output}=>critic_dataset::collect(serde_json::from_slice(&std::fs::read(config)?)?,&output)?,
        Command::Benchmark{config,output,steps,warmup,simulation_only,profile,train,cycles,no_artifacts,continuous_memory,trace}=>{
            anyhow::ensure!(!continuous_memory || !train,"continuous-memory is a fixed-policy collection diagnostic");
            anyhow::ensure!(!trace || (!train && !simulation_only && steps%128==0),"trace requires collection without PPO and complete128-step blocks");
            let config:learner::Config=serde_json::from_slice(&std::fs::read(config)?)?;std::fs::create_dir_all(&output)?;
            artifacts::atomic_json(&output.join("benchmark-config.json"),&config)?;
            let report=if train {learner::run(config,cycles,if no_artifacts{None}else{Some(output.as_path())},false,true)?}
                else {checks::benchmark(&config,steps,warmup,simulation_only,profile,continuous_memory,if trace{Some(output.as_path())}else{None})?};
            artifacts::atomic_json(&output.join("benchmark-result.json"),&report)?;report
        },
        Command::Replay{logs,output,r#match}=>{
            let root=PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
            let python=if cfg!(windows){"python"}else{"/home/zelt/dota-compat-check/bin/python"};
            let status=std::process::Command::new(python).arg(root.join("gpu/scripts/replay.py")).args(logs)
                .arg("--output").arg(&output).arg("--match").arg(r#match.to_string()).status()?;
            anyhow::ensure!(status.success(),"replay conversion failed");serde_json::json!({"replay":output})
        }
        Command::Stop{run}=>{artifacts::atomic_json(&run.join("stop.json"),&serde_json::json!({"reason":"graceful stop requested"}))?;serde_json::json!({"stop_requested":true})}
        Command::CellCheck=>checks::cell_gradient()?,
        Command::DumpInitialObservation{engine,output,ticks}=>checks::dump_initial_observation(&engine,&output,ticks)?,
        Command::DumpIdleLane{engine,reference,output,seconds}=>{
            let report=fidelity::idle_lane(&engine,&reference,seconds)?;artifacts::atomic_json(&output,&report)?;serde_json::json!({"diagnostic":output,"seconds":seconds})
        },
        Command::SimCheck{engine,matches,steps,inference}=>checks::simulator(&engine,matches,steps,inference)?,
        Command::TrainCheck{matches,cycles,batch}=>learner::train(learner::Config{matches,batch_sequences:batch,..Default::default()},cycles)?,
    };
    println!("{}",serde_json::to_string_pretty(&result)?);Ok(())
}
