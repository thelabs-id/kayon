use anyhow::Result;
use chrono::Utc;
use nvml_wrapper::Nvml;
use sysinfo::System;

use crate::ipc::*;

pub fn probe_machine() -> Result<MachineProfile> {
    let mut sys = System::new_all();
    sys.refresh_all();
    let gpus = probe_gpus();
    let primary_gpu_index = if gpus.is_empty() { None } else { Some(0) };
    let cpu = probe_cpu(&sys);
    let ram = probe_ram(&sys);
    let disks = probe_disks();
    let os = probe_os();

    Ok(MachineProfile {
        gpus,
        primary_gpu_index,
        cpu,
        ram,
        disks,
        os,
        probed_at: Utc::now(),
    })
}

fn probe_gpus() -> Vec<GpuInfo> {
    let nvml = match Nvml::init() {
        Ok(n) => n,
        Err(_) => return vec![],
    };

    let count = match nvml.device_count() {
        Ok(c) => c,
        Err(_) => return vec![],
    };

    let mut gpus = Vec::with_capacity(count as usize);
    for i in 0..count {
        let device = match nvml.device_by_index(i) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let name = device.name().unwrap_or_else(|_| "Unknown GPU".into());
        let pci = device.pci_info().ok();
        let pci_id = pci.as_ref().map(|p| {
            format!(
                "{:04x}:{:02x}:{:02x}.0",
                p.domain, p.bus, p.device
            )
        });
        let cc = device.cuda_compute_capability().ok();
        let compute_capability = cc.as_ref().map(|c| format!("{}.{}", c.major, c.minor));
        let arch = cc.and_then(|c| compute_arch(c.major, c.minor));

        let driver = nvml.sys_driver_version().ok();
        let cuda = nvml.sys_cuda_driver_version().ok().map(|v| {
            let major = v / 1000;
            let minor = (v % 1000) / 10;
            format!("{}.{}", major, minor)
        });

        let mem = device.memory_info().ok();
        let total_vram = mem.as_ref().map(|m| m.total).unwrap_or(0);

        let telemetry = GpuTelemetry {
            vram_used_bytes: mem.as_ref().map(|m| m.used).unwrap_or(0),
            vram_free_bytes: mem.as_ref().map(|m| m.free).unwrap_or(0),
            utilization_percent: device.utilization_rates().ok().map(|u| u.gpu as f32).unwrap_or(0.0),
            temperature_c: device
                .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                .ok()
                .map(|t| t as f32)
                .unwrap_or(0.0),
            power_watts: device.power_usage().ok().map(|p| p as f32 / 1000.0).unwrap_or(0.0),
            core_clock_mhz: device
                .clock_info(nvml_wrapper::enum_wrappers::device::Clock::Graphics)
                .ok()
                .unwrap_or(0),
            mem_clock_mhz: device
                .clock_info(nvml_wrapper::enum_wrappers::device::Clock::Memory)
                .ok()
                .unwrap_or(0),
        };

        gpus.push(GpuInfo {
            name,
            pci_id,
            architecture: arch,
            compute_capability,
            driver_version: driver,
            cuda_version: cuda,
            total_vram_bytes: total_vram,
            telemetry,
        });
    }
    gpus
}

fn compute_arch(major: i32, minor: i32) -> Option<String> {
    match (major, minor) {
        (5, _) => Some("Maxwell".into()),
        (6, _) => Some("Pascal".into()),
        (7, 0) => Some("Volta".into()),
        (7, _) => Some("Turing".into()),
        (8, 9) => Some("Ada Lovelace".into()),
        (8, _) => Some("Ampere".into()),
        (9, _) => Some("Hopper".into()),
        _ => None,
    }
}

fn probe_cpu(sys: &System) -> CpuInfo {
    let cpus = sys.cpus();
    let brand = cpus
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "Unknown CPU".into());
    let logical = cpus.len();
    // Physical cores vs logical threads differ on hyper-threaded CPUs (HW-3). Fall back to the
    // logical count only if the physical count is unavailable.
    let physical = sys.physical_core_count().unwrap_or(logical);
    let freq = cpus.first().map(|c| c.frequency()).unwrap_or(0);
    let usage = sys.global_cpu_usage();

    CpuInfo {
        brand,
        core_count: physical,
        thread_count: logical,
        frequency_mhz: freq,
        usage_percent: usage,
    }
}

fn probe_ram(sys: &System) -> RamInfo {
    let total = sys.total_memory();
    let available = sys.available_memory();
    RamInfo {
        total_bytes: total,
        available_bytes: available,
        used_bytes: total.saturating_sub(available),
    }
}

fn probe_disks() -> Vec<DiskInfo> {
    let mut disks = Vec::new();
    for disk in sysinfo::Disks::new_with_refreshed_list().list() {
        let mount = disk.mount_point().to_string_lossy().to_string();
        let total = disk.total_space();
        let available = disk.available_space();
        let kind = format!("{:?}", disk.kind());

        if total > 0 {
            disks.push(DiskInfo {
                mount,
                total_bytes: total,
                free_bytes: available,
                kind,
            });
        }
    }
    disks
}

fn probe_os() -> OsInfo {
    OsInfo {
        name: System::name().unwrap_or_else(|| "Unknown".into()),
        version: System::os_version().unwrap_or_else(|| "Unknown".into()),
        kernel_version: System::kernel_version(),
        host_name: System::host_name().unwrap_or_else(|| "Unknown".into()),
    }
}

pub fn poll_gpu_telemetry(gpu_index: u32) -> Option<GpuTelemetry> {
    let nvml = Nvml::init().ok()?;
    let device = nvml.device_by_index(gpu_index).ok()?;
    let mem = device.memory_info().ok()?;
    let util = device.utilization_rates().ok();
    let temp = device
        .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
        .ok();
    let power = device.power_usage().ok();
    let core = device
        .clock_info(nvml_wrapper::enum_wrappers::device::Clock::Graphics)
        .ok();
    let mem_clock = device
        .clock_info(nvml_wrapper::enum_wrappers::device::Clock::Memory)
        .ok();

    Some(GpuTelemetry {
        vram_used_bytes: mem.used,
        vram_free_bytes: mem.free,
        utilization_percent: util.map(|u| u.gpu as f32).unwrap_or(0.0),
        temperature_c: temp.map(|t| t as f32).unwrap_or(0.0),
        power_watts: power.map(|p| p as f32 / 1000.0).unwrap_or(0.0),
        core_clock_mhz: core.unwrap_or(0),
        mem_clock_mhz: mem_clock.unwrap_or(0),
    })
}

pub fn get_vram_free() -> u64 {
    let nvml = match Nvml::init() {
        Ok(n) => n,
        Err(_) => return 0,
    };
    nvml.device_by_index(0)
        .and_then(|d| d.memory_info())
        .map(|m| m.free)
        .unwrap_or(0)
}

pub fn get_vram_total() -> u64 {
    let nvml = match Nvml::init() {
        Ok(n) => n,
        Err(_) => return 0,
    };
    nvml.device_by_index(0)
        .and_then(|d| d.memory_info())
        .map(|m| m.total)
        .unwrap_or(0)
}
