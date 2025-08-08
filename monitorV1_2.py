import psutil, platform, socket, json, csv, datetime, os, time
from pathlib import Path
import pandas as pd
import plotly.express as px
import subprocess 
from tqdm import tqdm # 

# ✅ Variáveis de Configuração
server_name = socket.gethostname()

# 📁 Caminhos para armazenar logs e relatórios
IS_WINDOWS = platform.system() == 'Windows'
EXPORT_FOLDER = Path('C:/temp') if IS_WINDOWS else Path('/tmp')
os.makedirs(EXPORT_FOLDER, exist_ok=True)
csv_path = EXPORT_FOLDER / "disk_monitor.csv"
json_path = EXPORT_FOLDER / "disk_monitor.jsonl"
report_path = EXPORT_FOLDER / "disk_report.html"

# Define o tempo de espera entre as coletas (em segundos)
SLEEP_TIME = 3 

history = [] 

def get_system_info():
    server_type = "Físico"
    virtual_keywords = ['VMware', 'VirtualBox', 'KVM', 'Xen', 'Hyper-V', 'Virtual Machine', 'QEMU']

    try:
        if IS_WINDOWS:
            output = subprocess.check_output(['systeminfo'], universal_newlines=True, stderr=subprocess.DEVNULL)
            if any(keyword in output for keyword in virtual_keywords):
                server_type = "Virtual"
        else:
            output = subprocess.check_output(['sudo', 'dmidecode'], universal_newlines=True, stderr=subprocess.DEVNULL)
            if any(keyword in output for keyword in virtual_keywords):
                server_type = "Virtual"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    os_info = {
        'system': platform.system(),
        'version': platform.version(),
        'release': platform.release(),
    }
    
    disks_info = []
    partitions = psutil.disk_partitions(all=False)
    for p in partitions:
        if 'cdrom' in p.opts or p.fstype in ['vboxsf', 'vmhgfs']:
            continue
        if IS_WINDOWS and not p.mountpoint:
            continue
            
        disks_info.append({
            'device': p.device,
            'mountpoint': p.mountpoint
        })
        
    return {
        'server_type': server_type,
        'os_info': os_info,
        'disks_count': len(disks_info),
        'disks_info': disks_info
    }


def get_disk_snapshot():
    if IS_WINDOWS:
        path_to_check = 'C:\\'
    else:
        path_to_check = '/'
    
    usage = psutil.disk_usage(path_to_check)
    io_total = psutil.disk_io_counters()
    io_disks = psutil.disk_io_counters(perdisk=True)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    data = {
        'server': server_name,
        'timestamp': timestamp,
        'total': usage.total,
        'used': usage.used,
        'free': usage.free,
        'percent': usage.percent,
        'read_bytes': io_total.read_bytes,
        'write_bytes': io_total.write_bytes,
        'read_count': io_total.read_count,
        'write_count': io_total.write_count,
        'read_time': io_total.read_time,
        'write_time': io_total.write_time,
        'disks': {}
    }

    for name, stats in io_disks.items():
        data['disks'][name] = {
            'read_bytes': stats.read_bytes,
            'write_bytes': stats.write_bytes,
            'read_count': stats.read_count,
            'write_count': stats.write_count,
            'read_time': stats.read_time,
            'write_time': stats.write_time
        }
    return data

def export_logs(data):
    with open(json_path, 'a') as jf:
        jf.write(json.dumps(data) + "\n")
    
    row = {
        'server': data['server'],
        'server_type': data['server_info']['server_type'],
        'os_system': data['server_info']['os_info']['system'],
        'os_release': data['server_info']['os_info']['release'],
        'disks_count': data['server_info']['disks_count'],
        'timestamp': data['timestamp'],
        'total': data['total'],
        'used': data['used'],
        'free': data['free'],
        'percent': data['percent'],
        'read_bytes': data['read_bytes'],
        'write_bytes': data['write_bytes'],
        'read_count': data['read_count'],
        'write_count': data['write_count'],
        'read_time': data['read_time'],
        'write_time': data['write_time']
    }

    for disk, stats in data['disks'].items():
        for key, value in stats.items():
            row[f"{disk}_{key}"] = value

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def generate_html_report(history_data, current_snapshot, system_info):
    df = pd.DataFrame(history_data)
    
    io_fig = px.line(df, x='timestamp', y=['read_bytes', 'write_bytes'],
                     labels={'value': 'Bytes', 'variable': 'Operação'},
                     title=f'I/O do Disco - {current_snapshot["server"]}')

    system_info_html = f"""
    <div style="border: 1px solid #ccc; padding: 15px; margin-bottom: 20px;">
        <h3>Detalhes do Servidor</h3>
        <ul>
            <li><strong>Tipo de Servidor:</strong> {system_info['server_type']}</li>
            <li><strong>Sistema Operacional:</strong> {system_info['os_info']['system']} {system_info['os_info']['release']}</li>
            <li><strong>Total de Discos:</strong> {system_info['disks_count']}</li>
            <li><strong>Localização dos Discos:</strong> {[d['device'] for d in system_info['disks_info']]}</li>
        </ul>
    </div>
    """

    summary_html = '<h3>📋 Resumo do Monitoramento de Disco</h3>'
    summary_html += '<table style="width:100%; border-collapse: collapse;">'
    summary_html += '<thead><tr style="background-color:#f2f2f2;"><th>Disco</th><th>Leitura (KB)</th><th>Escrita (KB)</th><th>Contagem I/O</th><th>Latência (ms)</th></tr></thead>'
    summary_html += '<tbody>'
    for disk, stats in current_snapshot['disks'].items():
        read_kb = stats['read_bytes'] // 1024
        write_kb = stats['write_bytes'] // 1024
        io_count = stats['read_count'] + stats['write_count']
        latency = round((stats['read_time'] + stats['write_time']) / 2, 2)
        summary_html += f'<tr><td>{disk}</td><td>{read_kb}</td><td>{write_kb}</td><td>{io_count}</td><td>{latency}</td></tr>'
    summary_html += '</tbody></table>'
    
    html_content = f"""
    <html>
    <head>
        <title>Monitoramento de Disco - {current_snapshot['server']}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            h1, h3 {{ color: #333; }}
            table {{ border: 1px solid #ccc; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <h1>🛡️ Relatório de Desempenho de Disco</h1>
        <p><strong>Última atualização:</strong> {current_snapshot['timestamp']}</p>
        {system_info_html}
        <p><strong>Uso do Disco:</strong> {current_snapshot['percent']}%</p>
        {io_fig.to_html(full_html=False)}
        {summary_html}
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)

if __name__ == '__main__':
    try:
        duration_min = int(input("Por quantos minutos você deseja monitorar o disco? (Digite um número): "))
        if duration_min <= 0:
            print("A duração deve ser um número inteiro positivo. O monitoramento será encerrado.")
            exit()
    except ValueError:
        print("Entrada inválida. O monitoramento será encerrado.")
        exit()

    system_info = get_system_info()
    total_iterations = (duration_min * 60) // SLEEP_TIME

    print(f"\nMonitoramento de disco iniciado no servidor: {server_name}")
    print(f"Tipo de Servidor: {system_info['server_type']}")
    print(f"Sistema Operacional: {system_info['os_info']['system']} {system_info['os_info']['release']}")
    print(f"Total de Discos: {system_info['disks_count']}")
    print(f"Localização dos Discos: {[d['device'] for d in system_info['disks_info']]}")
    print(f"\nOs logs serão salvos na pasta: {EXPORT_FOLDER}")
    print(f"O relatório em HTML será gerado em: {report_path}")
    print(f"\nColetando dados a cada {SLEEP_TIME} segundos...")
    print("Pressione Ctrl+C para interromper o monitoramento.")

    try:
        # 🆕 Loop com barra de progresso do tqdm
        for _ in tqdm(range(total_iterations), desc="Monitoramento em andamento", unit="coleta"):
            snapshot = get_disk_snapshot()
            
            snapshot['server_info'] = system_info
            
            history.append(snapshot)
            if len(history) > 60: history.pop(0)
            
            export_logs(snapshot)
            generate_html_report(history, snapshot, system_info)
            
            time.sleep(SLEEP_TIME)
            
    except KeyboardInterrupt:
        tqdm.write("\nMonitoramento interrompido pelo usuário.")
        
    finally:
        # Mensagem final em destaque
        print("\n" + "="*80)
        print(">>> ✅ MONITORAMENTO CONCLUÍDO! <<<".center(80))
        print("="*80)
        print("\nPara verificar os dados coletados, acesse os arquivos gerados no local:")
        print(f"    📁 Caminho: {EXPORT_FOLDER}")
        print("\n    - Relatório visual: disk_report.html")
        print("    - Log de dados: disk_monitor.csv")
        print("    - Log de dados (JSON): disk_monitor.jsonl")
        print("\n")