#!/usr/bin/env node

/**
 * GeoAuPredict Development Server with Dashboards
 * 
 * This script automatically starts:
 * 1. Streamlit Dashboard (port 8501)
 * 2. Dash 3D Dashboard (port 8050)
 * 3. Next.js Dev Server (port 3000)
 * 
 * All processes are managed together and stop when you press Ctrl+C
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  yellow: '\x1b[33m',
  green: '\x1b[32m',
  blue: '\x1b[34m',
  red: '\x1b[31m',
  cyan: '\x1b[36m'
};

const processes = [];
const projectRoot = path.resolve(__dirname, '..');
const venvPath = path.join(projectRoot, 'venv', 'bin');

// Ensure logs directory exists
const logsDir = path.join(projectRoot, 'logs');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}

// Log file paths
const streamlitLog = path.join(logsDir, 'streamlit.log');
const dashLog = path.join(logsDir, 'dash.log');

function log(message, color = colors.reset) {
  console.log(`${color}${message}${colors.reset}`);
}

function startProcess(name, command, args, options = {}) {
  return new Promise((resolve, reject) => {
    log(`🚀 Starting ${name}...`, colors.cyan);
    
    const proc = spawn(command, args, {
      cwd: projectRoot,
      stdio: options.stdio || 'pipe',
      shell: true,
      env: {
        ...process.env,
        PATH: `${venvPath}:${process.env.PATH}`,
        ...options.env
      }
    });

    processes.push({ name, proc });

    if (proc.stdout) {
      proc.stdout.on('data', (data) => {
        const lines = data.toString().split('\n').filter(l => l.trim());
        lines.forEach(line => {
          log(`[${name}] ${line}`, options.color || colors.reset);
        });
      });
    }

    if (proc.stderr) {
      proc.stderr.on('data', (data) => {
        const lines = data.toString().split('\n').filter(l => l.trim());
        lines.forEach(line => {
          // Filter out common warnings
          if (!line.includes('DeprecationWarning') && 
              !line.includes('scatter_mapbox is deprecated')) {
            log(`[${name}] ${line}`, colors.yellow);
          }
        });
      });
    }

    proc.on('error', (error) => {
      log(`❌ Error starting ${name}: ${error.message}`, colors.red);
      reject(error);
    });

    proc.on('exit', (code, signal) => {
      if (code !== 0 && code !== null && signal !== 'SIGTERM' && signal !== 'SIGINT') {
        log(`⚠️  ${name} exited with code ${code}`, colors.yellow);
      }
    });

    // Give process a moment to start
    setTimeout(() => {
      if (!proc.killed) {
        log(`✅ ${name} started`, colors.green);
        resolve(proc);
      } else {
        reject(new Error(`${name} failed to start`));
      }
    }, options.startupDelay || 2000);
  });
}

async function checkPythonDeps() {
  return new Promise((resolve) => {
    log('📦 Checking Python dependencies...', colors.blue);
    
    const pip = spawn('pip', ['install', '-q', '-r', 'web_requirements.txt'], {
      cwd: projectRoot,
      env: {
        ...process.env,
        PATH: `${venvPath}:${process.env.PATH}`
      }
    });

    pip.on('close', (code) => {
      if (code === 0) {
        log('✅ Python dependencies ready', colors.green);
      } else {
        log('⚠️  Warning: Could not install Python dependencies', colors.yellow);
      }
      resolve();
    });
  });
}

function killAllProcesses() {
  log('\n🛑 Stopping all services...', colors.yellow);
  
  processes.forEach(({ name, proc }) => {
    if (!proc.killed) {
      log(`   Stopping ${name}...`, colors.yellow);
      proc.kill('SIGTERM');
    }
  });

  // Force kill after 5 seconds
  setTimeout(() => {
    processes.forEach(({ name, proc }) => {
      if (!proc.killed) {
        log(`   Force stopping ${name}...`, colors.red);
        proc.kill('SIGKILL');
      }
    });
    process.exit(0);
  }, 5000);
}

// Handle Ctrl+C
process.on('SIGINT', () => {
  killAllProcesses();
});

process.on('SIGTERM', () => {
  killAllProcesses();
});

async function main() {
  log('\n' + '='.repeat(60), colors.bright);
  log('🌟  GeoAuPredict Development Environment', colors.bright + colors.green);
  log('='.repeat(60) + '\n', colors.bright);

  try {
    // Check Python dependencies
    await checkPythonDeps();

    log('\n📍 Starting services...\n', colors.bright);

    // Start Streamlit
    await startProcess(
      'Streamlit',
      'streamlit',
      [
        'run',
        'src/app/spatial_validation_dashboard.py',
        '--server.port=8501',
        '--server.headless=true',
        '--browser.gatherUsageStats=false'
      ],
      {
        color: colors.cyan,
        startupDelay: 3000,
        env: { STREAMLIT_SERVER_HEADLESS: 'true' }
      }
    );

    // Start Dash
    await startProcess(
      'Dash',
      'python',
      ['src/app/3d_visualization_dashboard.py'],
      {
        color: colors.blue,
        startupDelay: 3000
      }
    );

    // Start Next.js
    await startProcess(
      'Next.js',
      'npm',
      ['run', 'dev'],
      {
        stdio: 'inherit', // Show Next.js output directly
        startupDelay: 2000
      }
    );

    log('\n' + '='.repeat(60), colors.bright);
    log('✅  All services running!', colors.bright + colors.green);
    log('='.repeat(60), colors.bright);
    log('\n📊 Access your application:', colors.bright);
    log('   • Main App:      http://localhost:3000', colors.cyan);
    log('   • Dashboards:    http://localhost:3000/dashboards', colors.cyan);
    log('   • Streamlit:     http://localhost:8501', colors.blue);
    log('   • Dash 3D:       http://localhost:8050', colors.blue);
    log('\n📋 Logs:', colors.bright);
    log(`   • Streamlit:     ${streamlitLog}`, colors.yellow);
    log(`   • Dash:          ${dashLog}`, colors.yellow);
    log('\n🛑 Press Ctrl+C to stop all services\n', colors.red);

  } catch (error) {
    log(`\n❌ Failed to start services: ${error.message}`, colors.red);
    killAllProcesses();
    process.exit(1);
  }
}

// Run main function
main();

