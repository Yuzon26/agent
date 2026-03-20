## env文件配置
开发阶段，每个人的配置可能不一样，为了方便各自开发，前端和后端的env文件都在.gitignore中，不进行git，因此clone本项目之后请自行复制前端和后端的.env.example文件到前端和后端的文件节中，防止产生不必要的报错。

## dist文件配置
为了防止两个开发人员同时改了前端，各自打包，产生的 dist 文件会产生海量冲突，dist文件夹不进行git

开发人员自行打包dist文件，步骤如下：
前端代码在frontend之中，如果对前端代码进行了修改，在frontend/DsAgentChat_web目录下运行
npm run build
npm run dev
然后将生成的dist文件夹复制到backend/deepseek_agent/llm_backend/static文件夹下

## requirement文件更新
补充了新的依赖库