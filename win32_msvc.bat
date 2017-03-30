@echo off

rem Variables to set.
set RELEASE=false

rem Warnings to ignore.
set COMMON_WARNINGS=-wd4189 -wd4706 -wd4996 -wd4100 -wd4127 -wd4267 -wd4505 -wd4820 -wd4365 -wd4514 -wd4062 -wd4061 -wd4668 -wd4389 -wd4018 -wd4711 -wd4987 -wd4710 -wd4625 -wd4626 -wd4350 -wd4826 -wd4640 -wd4571 -wd4986 -wd4388 -wd4129 -wd4201 -wd4623 -wd4774
 
IF NOT EXIST "build" mkdir "build"
set DEBUG_COMMON_COMPILER_FLAGS=-nologo -MTd -Gm- -GR- -EHsc- -Od -Oi %COMMON_WARNINGS% -DERROR_LOGGING=1 -DINTERNAL=1 -DMEM_CHECK=1 -DWIN32=1 -DLINUX=0 -FC -Zi -GS- -Gs9999999
set RELEASE_COMMON_COMPILER_FLAGS=-nologo -MT -fp:fast -Gm- -GR- -EHa- -O2 -Oi %COMMON_WARNINGS% -DERROR_LOGGING=0 -DRUN_TESTS=0 -DINTERNAL=0 -DMEM_CHECK=0 -DWIN32=1 -DLINUX=0 -FC -Zi -GS- -Gs9999999

rem Build code.
set FILES="../code/unsharp_mask.cpp"
pushd "build"
if "%RELEASE%"=="true" (
    cl -FeBlurDemo %RELEASE_COMMON_COMPILER_FLAGS% -Wall %FILES% -link -subsystem:console,5.2 kernel32.lib opencl.lib
) else (
    cl -FeBlurDemo %DEBUG_COMMON_COMPILER_FLAGS% -Wall %FILES% -link -subsystem:console,5.2 kernel32.lib opencl.lib
)
popd
