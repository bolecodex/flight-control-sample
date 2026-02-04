
文档中心
扣子
豆包语音
火山方舟大模型服务平台
API网关
云服务器
扣子
豆包语音
火山方舟大模型服务平台
API网关
云服务器
文档
备案
控制台
z
zhaoweibo.0820 / eps_yxd_group
账号管理
账号ID : 2108323502
联邦登陆
企业认证
费用中心
可用余额¥ 0.00
充值汇款
账户总览
账单详情
费用分析
发票管理
权限与安全
安全设置
访问控制
操作审计
API 访问密钥
工具与其他
公测申请
资源管理
配额中心
伙伴控制台
待办事项
待支付
0
待续费
0
待处理工单
0
未读消息
0
火山方舟大模型服务平台
文档指南
API参考
资源
请输入

文档首页

火山方舟大模型服务平台

Responses API

创建模型响应

复制全文

我的收藏
创建模型响应
 POST https://ark.cn-beijing.volces.com/api/v3/responses ​
本文介绍 Responses API 创建模型请求时的输入输出参数，供您使用接口时查阅字段含义。​
Tips：一键展开折叠，快速检索内容​
​
​
鉴权说明
快速入口
​
本接口支持 API Key /Access Key 鉴权，详见鉴权认证方式。​
​
​
​
请求参数​
跳转 响应参数​
请求体​
​
​
model string 必选​
您需要调用的模型的 ID （Model ID），开通模型服务，并查询 Model ID 。支持的模型请参见 模型列表。​
当您有多个应用调用模型服务或更细粒度权限管理，可通过 Endpoint ID 调用模型。​
​
​
input  string / array 必选​
输入的内容，模型需要处理的输入信息。​
信息类型​
​
​
​
instructions string / null ​
在模型上下文中插入系统消息或者开发者作为第一条指令。当与 previous_response_id 一起使用时，前一个回复中的指令不会被继承到下一个回复中。这样可以方便地在新的回复中替换系统（或开发者）消息。​
不可与缓存能力一起使用。配置了instructions 字段后，本轮请求无法写入缓存和使用缓存，表现为：​
caching 字段配置为 {"type":"enabled"} 时报错。​
传入带缓存的 previous_response_id 时，缓存输入（cached_tokens）为0。​
​
​
previous_response_id string / null ​
上一个模型回复的唯一标识符。使用该标识符可以实现多轮对话。​
note：在多轮连续对话中，建议在每次请求之间加入约 100 毫秒的延迟，否则可能会导致调用失败。​
​
​
expire_at integer 默认值：创建时刻+259200 ​
取值范围：(创建时刻, 创建时刻+604800]，即最多保留7天。​
设置存储的过期时刻，需传入 UTC Unix 时间戳（单位：秒），对 store（上下文存储） 和 caching（上下文缓存） 都生效。详细配置及示例代码说明请参见文档。​
注意：缓存存储时间计费，过期时刻-创建时刻 ，不满 1 小时按 1 小时计算。​
​
​
max_output_tokens integer / null ​
模型输出最大 token 数，包含模型回答和思维链内容。​
​
​
thinking object 默认值：取决于调用的模型 ​
控制模型是否开启深度思考模式。默认开启深度思考模式，可以手动关闭。​
属性​
​
​
​
reasoning object 默认值 {"effort": "medium"}​
限制深度思考的工作量。减少深度思考工作量可使响应速度更快，并且深度思考的 token 用量更小。​
属性​
​
​
​
caching object 默认值 {"type": "disabled"}​
是否开启缓存，阅读文档，了解缓存的具体使用方式。​
不可与 instructions 字段、tools（除自定义函数 Function Calling 外）字段一起使用。​
属性​
​
​
​
store boolean / null 默认值 true​
是否储存生成的模型响应，以便后续通过 API 检索。详细上下文管理使用说明，请见文档。​
false：不储存，对话内容不能被后续的 API 检索到。​
true：储存当前模型响应，对话内容能被后续的 API 检索到。​
​
​
stream boolean / null 默认值 false​
响应内容是否流式返回。流式输出示例见文档。​
false：模型生成完所有内容后一次性返回结果。​
true：按 SSE 协议逐块返回模型生成内容，并以一条 data: [DONE]消息结束。​
​
​
temperature float / null 默认值 1​
取值范围： [0, 2]。​
采样温度。控制了生成文本时对每个候选词的概率分布进行平滑的程度。当取值为 0 时模型仅考虑对数概率最大的一个 token。​
较高的值（如 0.8）会使输出更加随机，而较低的值（如 0.2）会使输出更加集中确定。​
通常建议仅调整 temperature 或 top_p 其中之一，不建议两者都修改。​
​
​
top_p float / null 默认值 0.7​
取值范围： [0, 1]。​
核采样概率阈值。模型会考虑概率质量在 top_p 内的 token 结果。当取值为 0 时模型仅考虑对数概率最大的一个 token。​
 0.1 意味着只考虑概率质量最高的前 10% 的 token，取值越大生成的随机性越高，取值越低生成的确定性越高。通常建议仅调整 temperature 或 top_p 其中之一，不建议两者都修改。​
​
​
text object​
模型文本输出的格式定义，可以是自然语言，也可以是结构化的 JSON 数据。详情请看结构化输出。​
属性​
​
tools array​
模型可以调用的工具，当您需要让模型调用工具时，需要配置该结构体。​
工具类型​
​
​
当前支持多种调用方式，包括​
内置工具（Built-in tools）：由方舟提供的预置工具，用以扩展模型内容，如豆包助手、联网搜索工具、图像处理工具、私域知识库搜索工具等。​
MCP工具：通过自定义 MCP 服务器与第三方系统集成。​
自定义工具（Function Calling）：您自定义的函数，使模型能够使用强类型参数和输出调用您自己的代码，使用示例见 文档 。​
豆包助手​
使用豆包助手，快速集成豆包app同款AI能力。详情请参考 豆包助手文档。​
注意：使用前需开通“豆包助手”功能。​
​
​
tools.type string 必选​
工具类型，此处填写工具名称，应为doubao_app。​
​
​
tools.feature object ​
豆包助手子功能。​
tools.feature.chat object​
日常沟通功能，豆包同款自由对话，默认关闭。​
tools.feature.chat.type string 默认值disabled​
取值范围：enabled， disabled。​
enabled：开启此功能。​
disabled：关闭此功能。​
​
tools.feature.chat.role_description string 默认值：你的名字是豆包,有很强的专业性。​
使用豆包助手时修改角色设定。​
此字段与system prompt、instructions 互斥。​
​
​
tools.feature.deep_chat object​
深度沟通功能，豆包同款深度思考对话，默认关闭。​
tools.feature.deep_chat.type string 默认值disabled​
取值范围：enabled， disabled。​
enabled：开启此功能。​
disabled：关闭此功能。​
​
tools.feature.deep_chat.role_description string 默认值：你的名字是豆包,有很强的专业性。​
使用豆包助手时修改角色设定。​
此字段与system prompt、instructions 互斥。​
​
​
​
tools.feature.ai_search object​
联网搜索功能，豆包同款AI搜索能力，默认关闭。​
tools.feature.ai_search.type string 默认值 disabled​
取值范围：enabled， disabled。​
enabled：开启此功能。​
disabled：关闭此功能。​
​
tools.feature.ai_search.role_description string 默认值：你的名字是豆包,有很强的专业性。​
使用豆包助手时修改角色设定。​
此字段与system prompt、instructions 互斥。​
​
​
tools.feature.reasoning_search object​
边想边搜功能，豆包同款结合思考过程的智能搜索能力，默认关闭。​
tools.feature.reasoning_search.type string 默认值 disabled​
取值范围：enabled， disabled。​
enabled：开启此功能。​
disabled：关闭此功能。​
​
tools.feature.reasoning_search.role_description string 默认值：你的名字是豆包,有很强的专业性。​
使用豆包助手时修改角色设定。​
此字段与system prompt、instructions 互斥。​
​
​
​
​
tools.user_location object 默认值{"type": "approximate"}​
用户地理位置，用于优化对话与搜索结果，包含 type、country、city、region 字段。示例如下：​
​
"user_location":{​
     "type":"approximate",​
     "country": "中国",​
     "region":"浙江",​
     "city":"杭州"​
}​
​
注意：填写 type 后，country、city、region 中 至少1个字段有有效值。​
​
Function Calling​
​
联网搜索工具​
​
图像处理工具​
​
MCP 工具​
​
私域知识库搜索工具​
​
​
tool_choice string / object​
仅 doubao-seed-1-6-*** 模型支持此字段。​
本次请求，模型返回信息中是否有待调用的工具。​
当没有指定工具时，none 是默认值。如果存在工具，则 auto 是默认值。​
可选类型​
​
max_tool_calls  integer  ​
取值范围： [1, 10]。​
最大工具调用轮次（一轮里不限制次数）。在工具调用达到此限制次数后，提示模型停止更多工具调用并进行回答。​
注意：该参数为尽力而为（best effort）机制，不保证成功，最终调用次数会受模型推理效果、工具返回结果有效性等因素影响。​
豆包助手不支持此参数。​
Web Search 基础联网搜索工具的默认值 3。​
Image Process 图像处理工具的默认值 10，不支持修改。​
Knowledge Search 私域知识库搜索工具的默认值为3。​
context_management  object  ​
上下文管理策略，帮助模型有效利用上下文窗口。​
属性​
​
响应参数​
跳转 请求参数​
非流式调用返回​
返回一个 response object。​
流式调用返回​
服务器会在生成 Response 的过程中，通过 Server-Sent Events（SSE）实时向客户端推送事件。具体事件介绍请参见 流式响应。​
​
​
单轮对话
多轮对话
上下文缓存
函数调用 Function Calling
request

import os
from openai import OpenAI

# 从环境变量中获取您的API KEY，配置方法见：https://www.volcengine.com/docs/82379/1399008
api_key = os.getenv('ARK_API_KEY')

client = OpenAI(
    base_url='https://ark.cn-beijing.volces.com/api/v3',
    api_key=api_key,
)

response = client.responses.create(
    model="doubao-seed-1-6-250615",
    input=[
            {
             "role": "system", 
             "content": "你是三字经小能手。每次用户输入时，你只能用三个汉字作出回应。用户输入如果是三个字，就用三个字像对对联一样进行匹配回应；如果不是三个字，就将用户输入的意思总结成三个字。无论何时，回复都严格限制为三个字。"
            },
            {
            "role": "user",
            "content":"人之初"
            }
          ],
    extra_body={
        "caching": {"type": "enabled"},
        "thinking":{"type":"disabled"}
    }
)
print(response)

second_response = client.responses.create(
    model="doubao-seed-1-6-250615",
    previous_response_id=response.id,
    input=[{"role": "user", "content": "下一句"}],
    extra_body={
        "caching": {"type": "enabled"},
        "thinking":{"type":"disabled"}
    }
)
print(second_response)

third_response = client.responses.create(
    model="doubao-seed-1-6-250615",
    previous_response_id=second_response.id,
    input=[{"role": "user", "content": "下一句"}],
    extra_body={
        "caching": {"type": "enabled"},
        "thinking":{"type":"disabled"}
    }
)
print(third_response)
response

[{
    "created_at": 1760168118,
    "id": "resp_0217*****",
    "max_output_tokens": 32768,
    "model": "doubao-seed-1-6-250615",
    "object": "response",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "性本善"
                }
            ],
            "status": "completed",
            "id": "msg_02176016811859200000000000000000000ffffac15aa1fde96cd"
        }
    ],
    "thinking": {
        "type": "disabled"
    },
    "service_tier": "default",
    "status": "completed",
    "usage": {
        "input_tokens": 101,
        "output_tokens": 3,
        "total_tokens": 104,
        "input_tokens_details": {
            "cached_tokens": 0
        },
        "output_tokens_details": {
            "reasoning_tokens": 0
        }
    },
    "caching": {
        "type": "enabled"
    },
    "store": true,
    "expire_at": 1760427318
},
{
    "created_at": 1760168211,
    "id": "resp_0217*****",
    "max_output_tokens": 32768,
    "model": "doubao-seed-1-6-250615",
    "object": "response",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "性相近"
                }
            ],
            "status": "completed",
            "id": "msg_02176016821111700000000000000000000ffffac15aa7b89be5a"
        }
    ],
    "previous_response_id": "resp_0217*****",
    "thinking": {
        "type": "disabled"
    },
    "service_tier": "default",
    "status": "completed",
    "usage": {
        "input_tokens": 116,
        "output_tokens": 2,
        "total_tokens": 118,
        "input_tokens_details": {
            "cached_tokens": 104
        },
        "output_tokens_details": {
            "reasoning_tokens": 0
        }
    },
    "caching": {
        "type": "enabled"
    },
    "store": true,
    "expire_at": 1760427411
},
{
    "created_at": 1760168277,
    "id": "resp_0217*****",
    "max_output_tokens": 32768,
    "model": "doubao-seed-1-6-250615",
    "object": "response",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "习相远"
                }
            ],
            "status": "completed",
            "id": "msg_02176016827706700000000000000000000ffffac15aa29c2672a"
        }
    ],
    "previous_response_id": "resp_0217*****",
    "thinking": {
        "type": "disabled"
    },
    "service_tier": "default",
    "status": "completed",
    "usage": {
        "input_tokens": 130,
        "output_tokens": 3,
        "total_tokens": 133,
        "input_tokens_details": {
            "cached_tokens": 118
        },
        "output_tokens_details": {
            "reasoning_tokens": 0
        }
    },
    "caching": {
        "type": "enabled"
    },
    "store": true,
    "expire_at": 1760427477
}]
request

import os
from openai import OpenAI

# 从环境变量中获取您的API KEY，配置方法见：https://www.volcengine.com/docs/82379/1399008
api_key = os.getenv('ARK_API_KEY')

client = OpenAI(
    base_url='https://ark.cn-beijing.volces.com/api/v3',
    api_key=api_key,
)

response = client.responses.create(
    model="doubao-seed-1-6-250615",
    input=[
            {
             "role": "system", 
             "content": "你是三字经小能手。每次用户输入时，你只能用三个汉字作出回应。用户输入如果是三个字，就用三个字像对对联一样进行匹配回应；如果不是三个字，就将用户输入的意思总结成三个字。无论何时，回复都严格限制为三个字。"
            },
            {
            "role": "user",
            "content":"人之初"
            }
          ],
    extra_body={
        "caching": {"type": "enabled"},
        "thinking":{"type":"disabled"}
    }
)
print(response)

second_response = client.responses.create(
    model="doubao-seed-1-6-250615",
    previous_response_id=response.id,
    input=[{"role": "user", "content": "下一句"}],
    extra_body={
        "caching": {"type": "enabled"},
        "thinking":{"type":"disabled"}
    }
)
print(second_response)

third_response = client.responses.create(
    model="doubao-seed-1-6-250615",
    previous_response_id=second_response.id,
    input=[{"role": "user", "content": "下一句"}],
    extra_body={
        "caching": {"type": "enabled"},
        "thinking":{"type":"disabled"}
    }
)
print(third_response)
response

[{
    "created_at": 1760168118,
    "id": "resp_0217*****",
    "max_output_tokens": 32768,
    "model": "doubao-seed-1-6-250615",
    "object": "response",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "性本善"
                }
            ],
            "status": "completed",
            "id": "msg_02176016811859200000000000000000000ffffac15aa1fde96cd"
        }
    ],
    "thinking": {
        "type": "disabled"
    },
    "service_tier": "default",
    "status": "completed",
    "usage": {
        "input_tokens": 101,
        "output_tokens": 3,
        "total_tokens": 104,
        "input_tokens_details": {
            "cached_tokens": 0
        },
        "output_tokens_details": {
            "reasoning_tokens": 0
        }
    },
    "caching": {
        "type": "enabled"
    },
    "store": true,
    "expire_at": 1760427318
},
{
    "created_at": 1760168211,
    "id": "resp_0217*****",
    "max_output_tokens": 32768,
    "model": "doubao-seed-1-6-250615",
    "object": "response",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "性相近"
                }
            ],
            "status": "completed",
            "id": "msg_02176016821111700000000000000000000ffffac15aa7b89be5a"
        }
    ],
    "previous_response_id": "resp_0217*****",
    "thinking": {
        "type": "disabled"
    },
    "service_tier": "default",
    "status": "completed",
    "usage": {
        "input_tokens": 116,
        "output_tokens": 2,
        "total_tokens": 118,
        "input_tokens_details": {
            "cached_tokens": 104
        },
        "output_tokens_details": {
            "reasoning_tokens": 0
        }
    },
    "caching": {
        "type": "enabled"
    },
    "store": true,
    "expire_at": 1760427411
},
{
    "created_at": 1760168277,
    "id": "resp_0217*****",
    "max_output_tokens": 32768,
    "model": "doubao-seed-1-6-250615",
    "object": "response",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "习相远"
                }
            ],
            "status": "completed",
            "id": "msg_02176016827706700000000000000000000ffffac15aa29c2672a"
        }
    ],
    "previous_response_id": "resp_0217*****",
    "thinking": {
        "type": "disabled"
    },
    "service_tier": "default",
    "status": "completed",
    "usage": {
        "input_tokens": 130,
        "output_tokens": 3,
        "total_tokens": 133,
        "input_tokens_details": {
            "cached_tokens": 118
        },
        "output_tokens_details": {
            "reasoning_tokens": 0
        }
    },
    "caching": {
        "type": "enabled"
    },
    "store": true,
    "expire_at": 1760427477
}]
最近更新时间：2026.01.23 14:16:39
这个页面对您有帮助吗？
有用
无用
上一篇：
对话(Chat) API
查询模型响应
下一篇


全天候售后服务
7x24小时专业工程师品质服务

极速服务应答
秒级应答为业务保驾护航

客户价值为先
从服务价值到创造客户价值

全方位安全保障
打造一朵“透明可信”的云
logo
关于我们
为什么选火山
文档中心
联系我们
人才招聘
云信任中心
友情链接
产品
云服务器
GPU云服务器
机器学习平台
客户数据平台 VeCDP
飞连
视频直播
全部产品
解决方案
汽车行业
金融行业
文娱行业
医疗健康行业
传媒行业
智慧文旅
大消费
服务与支持
备案服务
服务咨询
建议与反馈
廉洁舞弊举报
举报平台
联系我们
业务咨询：service@volcengine.com
市场合作：marketing@volcengine.com
电话：400-850-0030
地址：北京市海淀区北三环西路甲18号院大钟寺广场1号楼

微信公众号

抖音号

视频号
© 北京火山引擎科技有限公司 2025 版权所有
代理域名注册服务机构：新网数码 商中在线
服务条款
隐私政策
更多协议

京公网安备11010802032137号
京ICP备20018813号-3
营业执照
增值电信业务经营许可证京B2-20202418，A2.B1.B2-20202637
网络文化经营许可证：京网文（2023）4872-140号
