using Microsoft.AspNetCore.Hosting;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace GraduationWeb.Models
{
    public class Transfer
    {
        public async Task<string[]> transfer(Microsoft.AspNetCore.Http.IFormFile image, string style,string method)
        {
            string fileExt = image.FileName.Substring(image.FileName.LastIndexOf('.')); //文件扩展名，不含“.”
            long fileSize = image.Length; //获得文件大小，以字节为单位
            string newFileName = System.Guid.NewGuid().ToString() + fileExt; //随机生成新的文件名
            var filePath = @"D:/GraduationProject/Fast_style_transfer/upload/" + newFileName;
            using (var stream = new FileStream(filePath, FileMode.Create))
            {
                await image.CopyToAsync(stream);
            }
            string[] strArr = new string[3];
            strArr[0] = style;
            strArr[1] = method;
            strArr[2] = filePath;
            return RunPythonScript("transferWeb.py", "-u", strArr);
        }

        private string[] RunPythonScript(string sArgName, string args = "", params string[] teps)
        {
            Process p = new Process();
            //string path = System.AppDomain.CurrentDomain.SetupInformation.ApplicationBase + sArgName;// 获得python文件的绝对路径（将文件放在c#的debug文件夹中可以这样操作）
            string path = @"D:/GraduationProject/Fast_style_transfer/" + sArgName;//(因为我没放debug下，所以直接写的绝对路径,替换掉上面的路径了)
            //p.StartInfo.FileName = @"D:\Python\envs\python3\python.exe";//没有配环境变量的话，可以像我这样写python.exe的绝对路径。如果配了，直接写"python.exe"即可
            p.StartInfo.FileName = "python3.6.exe";
            string sArguments = path;
            foreach (string sigstr in teps)
            {
                sArguments += " " + sigstr;//传递参数
            }

            sArguments += " " + args;

            p.StartInfo.Arguments = sArguments;
            p.StartInfo.UseShellExecute = false;
            p.StartInfo.RedirectStandardOutput = true;
            p.StartInfo.RedirectStandardInput = true;
            p.StartInfo.RedirectStandardError = true;
            p.StartInfo.CreateNoWindow = true;

            p.Start();

            StreamReader reader = p.StandardOutput;//截取输出流
            string line = "";//每次读取一行
            string[] result = new string[2];
            while (!reader.EndOfStream)
            {
                line = reader.ReadLine();
                Console.WriteLine(line);
                if (line.Contains("result:"))
                    result[0] = line.Substring(7);
                if (line.Contains("time consumption:"))
                    result[1] = line.Substring(17);
            }

            p.BeginErrorReadLine();
            p.ErrorDataReceived += new DataReceivedEventHandler(p_OutputDataReceived);
            /*
            p.BeginOutputReadLine();
            p.OutputDataReceived += new DataReceivedEventHandler(p_OutputDataReceived);
            */

            p.WaitForExit();
            p.Close();

            return result;
        }

        //输出打印的信息
        static void p_OutputDataReceived(object sender, DataReceivedEventArgs e)
        {
            if (!string.IsNullOrEmpty(e.Data))
            {
                Console.WriteLine(e.Data + Environment.NewLine);
            }

        }
    }
}
