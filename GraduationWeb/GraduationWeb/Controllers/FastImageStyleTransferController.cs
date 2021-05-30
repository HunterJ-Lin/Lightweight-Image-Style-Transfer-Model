using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using GraduationWeb.Models;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace GraduationWeb.Controllers
{
    public class FastImageStyleTransferController : Controller
    {

        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> StyleTransfer(IFormFile image,string style,string method)
        {
            string[] result = await new Transfer().transfer(image, style, method);
            return Json(new { file=result[0],time=result[1] });
        }

        [HttpPost]
        public async Task<IActionResult> Train(IFormFile image, string method)
        {
            string result = new Train().train(image, method);
            return Json(new { status=result });
        }

        [HttpGet]
        public IActionResult GetMethodlList()
        {
            List<string> methods = new List<string>();
            methods.Add("complex");
            methods.Add("scale");
            methods.Add("fire");
            methods.Add("normal");
            return Json(new { list=methods});
        }


        [HttpGet]
        public IActionResult GetModelList()
        {
            Dictionary<string, List<string>> dic = new Dictionary<string, List<string>>();
            List<string> normal = new List<string>();
            List<string> fire = new List<string>();
            List<string> scale = new List<string>();
            List<string> complex = new List<string>();
            string path = @"/styleimage/";
            normal.Add(path + "cubist.jpg");
            normal.Add(path + "lamuse.jpg");
            normal.Add(path + "mosaic.jpg");
            normal.Add(path + "sm.jpg");
            normal.Add(path + "starry_night.jpg");
            normal.Add(path + "udnie.jpg");
            normal.Add(path + "wave.jpg");
            normal.Add(path + "feathers.jpg");
            normal.Add(path + "cuphead.jpg");
            normal.Add(path + "scream.jpg");
            fire.Add(path + "starry_night.jpg");
            fire.Add(path + "cuphead.jpg");
            fire.Add(path + "mosaic.jpg");
            scale.Add(path + "lamuse.jpg");
            scale.Add(path + "starry_night.jpg");
            scale.Add(path + "udnie.jpg");
            scale.Add(path + "feathers.jpg");
            complex.Add(path + "scream.jpg");
            complex.Add(path + "starry_night.jpg");
            dic.Add("normal", normal);
            dic.Add("fire", fire);
            dic.Add("scale", scale);
            dic.Add("complex", complex);
            return Json(new { dictionary=dic});
        }
    }
}