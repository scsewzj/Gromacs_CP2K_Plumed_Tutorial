# Gromacs_CP2K_Plumed_Tutorial
This is a tutorial for cross compilation of Gromacs and CP2K with the support of Plumed, CUDA and MPI. It is possibly the lastest combination of Gromacs and CP2K. Wish you a nice MD and QMMM journey!



> 这段提示区就说明适合看这篇博客的场景吧。如果是单机仅需要单独使用CP2K，您可以考虑直接安装cp2k的psmp包，或者按照文档的toolchain部分解决；如果是单独使用gromacs的，我会推荐使用conda的conda-forge channel里的包，注意选择构建时cuda或者mpi的依赖。后续内容主要是面向cp2k和gromacs交叉编译的。

@[TOC](至 Gromacs 2024.4，与CP2K的交叉编译以实现QMMM的MD计算)

---
首先这篇博客参考了计算化学公社的[Sob](http://bbs.keinsci.com/thread-21608-1-1.html)老师、[wuzhiyi](http://bbs.keinsci.com/thread-21783-1-1.html)老师，知乎的[supernova](https://zhuanlan.zhihu.com/p/700073814)老师以及bioexcel的[Dmitry Morozov](https://bioexcel.eu/webinar-multiscale-qm-mm-simulations-exploring-chemical-reactions-using-novel-gromacs-cp2k-interface-2020-12-08/)老师的方案，感谢他们博客和视频对我搭建和撰写这篇博客的帮助，希望我的博客可以帮助到其他后来者。(点前面蓝色的老师名字可以查看他们的原始文章)
# 版本问题
`CP2K: 2025.2及以前`
`Gromacs: 2026以前（不含）`
`Plumed: 无明确限制，但是个人建议10.0或9.3，根据gromacs`

我构建的几个需求主要是：
* 支持MPI (最少是单节点，虽然单节点未必需要mpi，tmpi或者就omp也行，但是我习惯有两级并行)
* 支持CUDA (因为有时候几百核跑MD不一定跑得过两张GPU，且我们的CPU用的人多，GPU用的少，或者没条件的你甚至可以用云计算的，毕竟CPU租赁看着便宜，对稍大一点体系按墙钟我个人经验还是GPU便宜)
* 支持Plumed（因为有同事和上级是化学出身，我本身做计算生物/生物信息的，我不能保证他们是否需要对计算场景需要扩展，所以支持Plumed对扩展计算场景是一种保障）
* 支持CP2K （单纯力学对于一些电荷或者弱相互作用有限，因为我的场景涉及计算化学），我的QM规模也可能在数百甚至到数千原子，所以我是希望支持XTB这种半经验的。那么2026的因为无法产生静态库，但XTB从2025.2才开始支持，所以我是只能使用2025.2，您要是不需要新特性，可能使用更早版本兼容性更好，但是2025.2是可以成功的。
* 且三者中Gromacs作为主控，我个人是觉得gromacs虽然一开始学习构建时候这个学习成本曲线比较陡峭，但是封装和扩展比计算化学的软件好一些。（而且我是先会Gromacs和ORCA的，有一定路径依赖，您如果有其他方案，十分欢迎您分享交流）

---

`提示：以下是本篇文章正文内容，下面案例可供参考`

# 一、Plumed安装
以下是wuzhiyi老师的方案，我原封不动搬运过来，这个方案对新的独立plumed依然是有效的，您注意修改版本和路径即可。如果您的架构不同，特别是弹性服务或者容器的环境，您的编译器可能需要手动指定或者您为这个环境单独构建。

`wuzhiyi老师的方案`
先安装plumed，测试了[2.7.2发行版](https://github.com/plumed/plumed2/releases/tag/v2.10.0)(点蓝字链接可以跳转，原始人拷贝粘贴https://github.com/plumed/plumed2/releases/tag/v2.10.0)， 理论上应该V2.7分支的都可以。
放在/Users/formidable/src/解压生成plumed-2.10.0文件夹，进入文件夹，先进行配置
```bash
然后直接用configure安装
```bash
./configure --prefix=/root/plumed/2.10.0 
make -j24
make install
```


用module的同学可以复制module文件方便设置环境变量
```bash
cp /usr/local/plumed/2.10.0/lib/plumed/modulefile ~/privatemodules/plumed/2.10.0
```

`新方案`

新方案就是不使用独立的plumed，转而采用cp2k自动安装的plumed，环境更干净，除非您需要单独的plumed。但cp2k安装的plumed除了版本早几个小版本，对一些软件的新版不兼容外，其他我的体验是没太大区别（因为很多新版特性用不上，加上不兼容问题多了，并不是很建议追求最新版，最好是24-25期间的版本，大多有较为成熟的解决方案）

其实我不知道wuzhiyi老师使用独立的plumed的原因，因为似乎很早版本的cp2k的toolchain就有plumed的依赖。但是岁月史书一下，可能跟早期他参照的版本并没有明确开plumed依赖有关，所以他plumed部分可能是参照的plumed的文档。

简单来说这一节几个大字就是`看cp2k部分，记得开--with-plumed=install`

# 二、CP2K安装
## cp2k安装
可以从[cp2k仓库](https://github.com/cp2k/cp2k)clone或者下载release的tarball（https://github.com/cp2k/cp2k 点蓝字跳转，原始人请复制）。这里面的库您可以按需要安装，但是我建议对大多数人
* `--with-plumed=system`，因为对于做科学计算的来说，openmpi大概率是系统已经有的，而且很可能您安装了ORCA等其他对mpi版本可能有限制的软件，您最好确认谁对版本较为宽容，cp2k通常要求不严格。并且您也可以选择其他系列的mpi，这个就需要看您的架构和偏好了。或者您也可以像我一样，因为受限于已经存在的mpi版本，所以专门让cp2k安装自己使用的mpi以避免冲突。
* `libxsmm`这个库比较魔幻，因为折腾了很久我也不敢说我完全清楚了这个库，应该是个小矩阵乘法运算的优化。我没有实测+/-的影响是否***，但是从功能上猜想应该不小，所以还是努力跑通，并没有删除标志让他回退到基本的blas。这个库你在cp2k中安装时是不会产生问题的，或许您使用cp2k的2024.1及以前的版本可能整个方案都不会有问题，至少从wuzhiyi老师和supernova老师的方案看起来是这样。但是如果您在最后尝试编译gromacs时候会出现缺失标志或者变量的问题，或者编译通过但是运行时报错。（supernova老师的博客提及了链接cp2k库时候他看到有其他人说删除标志解决，但是不知道是不是这个库。我这么猜想是因为我差点也想摆烂这么干。）
* `--enable-cuda=yes --gpu-ver=V100`这两个完全看您的设备，您如果是纯CPU设备或者GPU不在支持设备中，您可以考虑关闭，具体支持的设备您可以通过`./install_cp2k_toolchain.sh -h`查看

简单来说进入/cp2k-vx.x.x/tools/toolchain下
用toolchain脚本安装依赖，可以根据自己的需求增减。
```bash
./install_cp2k_toolchain.sh \
--with-libxsmm=install \
--with-elpa=install \
--with-libxc=install \
--with-libint=install \
--with-gsl=no \
--with-libvdwxc=no \
--with-spglib=no \
--with-hdf5=no \
--with-spfft=no \
--with-cosma=no  \
--with-libvori=no \
--with-sirius=no \
--with-scalapack=install \
--with-openblas=system \
--with-fftw=install \
--with-openmpi=install \
--with-plumed=install \
--enable-cuda=yes \
--gpu-ver=V100
```

然后将生成的依赖复制到根目录进行安装, 这个通常在您成功完成上一步后，脚本结尾会给您相关指示，就是拷贝tool的arch下内容到主目录的arch下
```bash
cp -r cp2k-vx.x.x/tools/toolchain/install/arch/* cp2k-vx.x.x/arch
source cp2k-vx.x.x/tools/toolchain/install/setup
```
根据[wuzhiyi老师说的sob老师说的](http://bbs.keinsci.com/thread-21608-1-1.html)(原始人请复制http://bbs.keinsci.com/thread-21608-1-1.html)（套个娃）一样，不过因为Gromacs要将cp2k作为库来调用，所以要编译cp2k库。
```bash
make ARCH=local VERSION="ssmp psmp" libcp2k
```
官方脚本会让您多编译出两个版本，主要是dbg的，如果没特别需要那就不需要了，当然您也可以蛋糕店里卖蛋糕，面包店里买面包。


# 三、Gromacs安装
通常即使您前序步骤信心满满地完成，觉得马上要成功了，我祝福这是真的。但有可能存在一些暗病，要继续集中精神直到测试成功。

最后我们进行[gromacs](https://gitlab.com/gromacs/gromacs/)的安装，从这里下载文件https://gitlab.com/gromacs/gromacs/-/tree/vx.x.x(点蓝字可以跳转，原始人请复制)。 

我建议使用2024.4版本，虽然2025版本绝大多数步骤都能完成，但！是绝大多数！且问题是plumed模块编译时候报变量缺失！经过我并不仔细地研究，对2025版本的gromacs，即使您使用Plumed最新的版本且里面可以选择2025.0的gromacs，其行为也和2024版本不同，虽然能够运行，但是似乎不会产生.h/.inc/.make的文件，我不能否定通过一些方式实现成功编译的可能，但是看起来不是很容易解决，所以我建议回退到2024.4（这是我能成功编译的）。当然您可以说‘嘿，我这暴脾气’，我也很喜欢您的暴脾气，希望您心脑血管健康。

下载后，解压进入gromacs的目录，使用cp2k下的plumed为gromacs打补丁

```
cp2k-vx.x.x/tools/toolchain/install/plumed-vx.x.x/bin/plumed-patch -p
#或者部分版本中命令是 
#plumed patch -p
#或者二者皆可
```	

然后选择相应版本的gromacs的序号即可，我记得最新支持到2025.0，但是您可以尝试用老plumed为新的gromacs打补丁，这有时是有效的（嗯，也是经过我不仔细的研究）。成功与否的指征是gromacs的根目录下是否有`Plumed.h/.inc/.cmake`。

接下来就可以执行最后的编译安装了

`特别说明`
很多人fftw应该不需要开`-DGMX_BUILD_OWN_FFTW=ON `，我是因为报了跟cp2k的精度兼容问题所以干脆让gmx编译自己的。BLAS和LAPACK您也可以使用cp2k的或者用包管理器安装的，只要库的路径正确。最好是全部使用静态链接，禁用动态链接，以免运行时指定了变量造成冲突。
```
mkdir build
cd build
cmake .. \
	-DBUILD_SHARED_LIBS=OFF \
     -DGMXAPI=OFF \
     -DGMX_INSTALL_NBLIB_API=OFF \
     -DGMX_GPU=CUDA \
     -DGMX_USE_COLVARS=internal \
     -DGMX_MPI=ON \
     -DGMX_FFT_LIBRARY=fftw3 \
     -DGMX_BUILD_OWN_FFTW=ON \
     -DGMX_BLAS_USER=/usr/local/lib/libopenblas.so \
     -DGMX_LAPACK_USER=/usr/local/lib/libopenblas.so \
     -DGMX_CP2K=ON \
     -DCP2K_DIR=/.../cp2k-2025.2/lib/local/psmp \
     -DCMAKE_INSTALL_PREFIX=/.../gromacs_cp2k_plumed/ \
     -DGMX_DEFAULT_SUFFIX=OFF \
     -DCP2K_LINKER_FLAGS="-L'/.../cp2k-2025.2/tools/toolchain/install/fftw-3.3.10/lib' \
    -L'/.../cp2k-2025.2/tools/toolchain/install/libint-v2.6.0-cp2k-lmax-5/lib' \
    -L'/.../cp2k-2025.2/tools/toolchain/install/libxc-7.0.0/lib' \
    -L'/.../cp2k-2025.2/tools/toolchain/install/scalapack-2.2.2/lib' \
    -L'/.../cp2k-2025.2/tools/toolchain/install/libxsmm-e0c4a2389afba36c453233ad7de07bd92c715bec/lib' \
    -L'/.../cp2k-2025.2/tools/toolchain/install/elpa-2024.05.001/cpu/lib/' \
    -L'/.../cp2k-2025.2/tools/toolchain/install/plumed-2.9.3/lib' \
    -L'/.../cp2k-2025.2/tools/toolchain/install/openmpi-5.0.8/lib' \
    -lstdc++ -lplumed -ldl -lz -lelpa_openmp -lscalapack -lxsmmf  -lxsmm -lpthread -lxcf03 -lxc -lint2 -lfftw3_mpi -lfftw3 -lfftw3_omp -lopenblas -Wl,--enable-new-dtags"
make -j24
make install
```
如果您幸运地成功了的话，您可能幸运地成功了！但人生不如意事十之八九，您或许会看到xsmm这个库相关的错误，比较典型的是找不到其库的库，这就都怪库库雷利亚了。如果您问库库雷利亚是谁，根据发型，可能是图0派老祖。

言归正传，如果只是找不到库，我相信您比较容易处理，您可以export或者在`-DCP2K_LINKER_FLAGS`中尝试指定，但其实上述配置中已经指定了，找不到的原因我无法下定论，但是export通常能解决。

比较棘手的是，您可能在编译时，或者侥幸通过编译但是运行时报libxsmm缺少一个带omp标志的变量，或者未定义。您可以尝试手动修改`/cp2k-vx.x.x/tools/toolchain/scripts/stage4`下安装`libxsmm`的脚本，在make时增加`-OMP=1`的标志。然后重新编译cp2k产生cp2k.a和psmp，然后再重复本节前序步骤编译安装gromacs。

如果您幸运地成功了的话，您可能幸运地成功了！但您或许还是会看到一样的报错，或者错误变成了缺少`xsmmext`，或者也可能是blas报omp相关错误，但通常omp有冲突时候也只是运行时报错，不会在编译安装的链接时候报错。此时的方案是您可以检查原来libxsmm的lib下，是否存在名称为libxsmmext.so的库，或许还有.so.1。那么您有可能离成功不远了，您接下来只需要将`-lxsmmext`参数加在`-DCP2K_LINKER_FLAGS`的标志中，即最后一行改成
```
-lstdc++ -lplumed -ldl -lz -lelpa_openmp -lscalapack -lxsmmf -lxsmmext -lxsmm -lpthread -lxcf03 -lxc -lint2 -lfftw3_mpi -lfftw3 -lfftw3_omp -lopenblas -Wl,--enable-new-dtags
```
然后重新编译安装gromacs，您大概率可以成功。

# 测试
最后我建议您随便生成一个水溶剂盒子，然后生成一个带有QM组的index文件进行一下简单的测试，体系的`.gro`和配置`.mdp`文件您可以让AI帮您生成，反正符不符合原理都无所谓，您只需要确认cp2k接口可以工作。您通常在`grompp`输出中可以观察到`QMMM Interface with CP2K is active, topology was modified!`字样，并且可以观察到`mdrun`后有`.inp`文件出现（这个就是gromacs生成的cp2k的输入文件）。对于plumed的验证比较简单，如果您成功开启了plumed的支持您运行`gmx`或`gmx_mpi` `-version`或`-h`就可以在输出中观察到版本号中存在plumed字样，如`:-) GROMACS - gmx grompp, 2024.4-plumed_2.9.3-dev (-:`（其他命令也行，对的，我从grompp粘来的，那当我前面没说，你随便运行个命令就行）。

# Trouble Shooting

其他需要注意的点是，如果您是一个比较常做科学计算或者AI的，您可能是有conda系环境的，请注意您当前的环境，且环境中是否有上述库，特别是cp2k中链接的。即使您编译时候指定也有可能被conda的变量覆盖造成一些变量undifined、reference的问题或者库依赖的库找不到。最保险的方案是开一个完全干净环境，特别注意blas、mkl以及如果您使用intelmpi，如果您默认有这些库且冲突请在编译时候禁用这些环境变量，或者可以配置为使用系统并提供路径到标志，但这一步需要在cp2k安装时就使用，并保持链接标志和cp2k实际使用的一致。

即使如此依然在运行时可能报找不到某些动态库，但这通常可以靠export解决，为了使用方便您可以将其写入您的.bashrc，如：
```
export LD_LIBRARY_PATH=/.../cp2k-2025.2/tools/toolchain/install/libxsmm-e0c4a2389afba36c453233ad7de07bd92c715bec/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/.../cp2k-2025.2/tools/toolchain/install/plumed-2.9.3/lib:$LD_LIBRARY_PATH
```

您如果有问题可以留言或者发私信与我讨论，感谢您的阅读！
最后祝愿您编译全绿，头顶不绿！

# 植入广告
你以为我要植入什么，小博主没货带，也没实验室要宣传，那就谢谢你这么帅/美还看我博客吧。

