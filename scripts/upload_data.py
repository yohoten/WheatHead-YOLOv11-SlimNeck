"""
数据打包和上传脚本 - 优云智算平台
===================================
用于将数据集打包并上传到优云智算平台存储

使用方法:
    python scripts/upload_data.py
"""

import os
import shutil
import tarfile
from pathlib import Path


def create_data_archive(source_dir, output_file):
    """
    创建数据压缩包
    
    Args:
        source_dir: 源目录路径
        output_file: 输出压缩文件路径
    """
    print(f"📦 正在打包数据: {source_dir}")
    
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    
    file_size = os.path.getsize(output_file) / (1024 ** 3)  # GB
    print(f"✅ 打包完成: {output_file} ({file_size:.2f} GB)")
    
    return output_file


def verify_data_structure(data_root):
    """
    验证数据结构是否正确
    
    Args:
        data_root: 数据根目录
    """
    print("\n🔍 验证数据结构...")
    
    required_dirs = [
        'images/train',
        'images/validation',
        'images/test',
        'labels/train',
        'labels/validation',
        'labels/test',
    ]
    
    missing = []
    for dir_path in required_dirs:
        full_path = Path(data_root) / dir_path
        if not full_path.exists():
            missing.append(dir_path)
    
    if missing:
        print("❌ 缺少以下目录:")
        for m in missing:
            print(f"   - {m}")
        return False
    
    # 统计文件数量
    train_images = len(list((Path(data_root) / 'images' / 'train').glob('*.jpg')))
    val_images = len(list((Path(data_root) / 'images' / 'validation').glob('*.jpg')))
    test_images = len(list((Path(data_root) / 'images' / 'test').glob('*.jpg')))
    
    print(f"✅ 数据结构验证通过")
    print(f"   训练集图片: {train_images} 张")
    print(f"   验证集图片: {val_images} 张")
    print(f"   测试集图片: {test_images} 张")
    
    return True


def main():
    """主函数"""
    data_root = Path('data/GlobalWheat2020')
    
    if not data_root.exists():
        print(f"❌ 数据目录不存在: {data_root}")
        print("请确保数据已下载到正确位置")
        return
    
    # 验证数据结构
    if not verify_data_structure(data_root):
        print("\n❌ 数据结构不完整，请检查")
        return
    
    # 创建压缩包
    output_file = 'data_archive.tar.gz'
    try:
        create_data_archive(data_root, output_file)
        
        print("\n" + "=" * 80)
        print("📤 下一步：上传数据到优云智算平台")
        print("=" * 80)
        print(f"\n方法 1: 使用优云 CLI 工具")
        print(f"  uyun upload {output_file} --bucket wheat-data")
        print(f"\n方法 2: 通过 Web 界面上传")
        print(f"  1. 登录优云智算平台")
        print(f"  2. 进入数据存储页面")
        print(f"  3. 上传 {output_file}")
        print(f"  4. 解压到指定目录")
        print(f"\n方法 3: 使用 scp/rsync")
        print(f"  scp {output_file} user@uyun-server:/path/to/data/")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 打包失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
