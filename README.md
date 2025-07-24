# Readme
该项目提供了一个MCP服务器，该服务器可以接受本地指定的人脸图像和证件图像，并交由指定的Comfyui服务器进行人脸与背景的融合，最终在本地保存融合的结果。
## 项目框架
```
├── README.md
├── requirements.txt
├── server.py MCP服务器
├── source_test.png 人脸图像示例
├── target_test.jpg 证照图像示例
├── 4.jpg 返回结果示例
└── mask_to_box_node.py 仿射变换模块
 ```
## 使用方法
###  1. 安装依赖
```
pip install -r requirements.txt
```
### 2. 安装Comfyui相关插件和模型
包含APersonMaskGenerator插件和使用到的flux-fill模型
### 3. 修改参数
1. 修改server_address为自己的MCP服务器地址
1. 修改mcp.run()中的MCP服务器端口和路径
2. 如有需要，可以适当修改仿射变换中的目标区域坐标dst_pts
### 4. 运行
```
python server.py
```
## MCP Client
以Claude为例，在claude_desktop_config.json文件中添加以下内容，替换为自己的MCP server地址
```
{
  "mcpServers": {
	"FLUX": {
            "command": "npx",
            "args": ["mcp-remote", "http://your mcp server address"]
        	}
	}
}
```
## 示例
提示词需包含三个参数，人脸图像路径，证照图像路径和保存文件夹路径。
```
source的路径为"D:\study\2025Autumn\code\source_test.jpg",target的文件路径为"D:\study\2025Autumn\code/target_test.jpg",帮我使用flux模型进行人脸与背景的融合，并保存到"D:\study\code\mayi\output"文件夹中
```
返回结果
## 待改进方向
1. 初次运行时由于要加载大模型，mcp server端运行时间较长，大模型端可能会报错 ```No result received from client-side tool execution.```(但最后成功保存结果没有问题)
2. flux模型填充效果 可以通过调整prompt等参数优化