# Third-party notices

The code under `cuda/` is a line-by-line port of RocketSim V3 and of the parts of Bullet Physics and glam it is built on,
plus a port of CORE-MATH's single-precision functions. `v3/fetch_v3.sh` downloads RocketSim V3 itself (it is not part of
this repository) and applies a patch to it for the tests.

## RocketSim V3

https://github.com/ZealanL/RocketSim (branch `v3-rust`, commit 45722876705adf5dc673aa764e80382cfe8c3312). MIT licence:

    MIT License
    
    Copyright (c) 2022 ZealanL
    Copyright (c) 2025 VirxEC
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

## Bullet Physics

RocketSim V3 is a Rust port of the parts of Bullet Physics that RocketSim uses (broadphase, narrowphase, constraint solver,
raycast vehicle); `cuda/sim/` ports that code again and carries its algorithms. Bullet is licensed under the zlib licence.
The V3 source tree carries no Bullet notice of its own; this is the notice of the Bullet Physics SDK
(https://github.com/bulletphysics/bullet3):

    Bullet Continuous Collision Detection and Physics Library
    Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org

    This software is provided 'as-is', without any express or implied warranty.
    In no event will the authors be held liable for any damages arising from the use of this software.
    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it freely,
    subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software.
       If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
    2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
    3. This notice may not be removed or altered from any source distribution.

## glam

https://github.com/bitshifter/glam-rs (version 0.34.1), MIT OR Apache-2.0. `cuda/common/glam.cuh` reproduces the arithmetic of
its f32 types operation by operation; the tests build against glam itself as the reference.

## CORE-MATH

https://core-math.gitlabpages.inria.fr/ (git 0ce0cbc15935e24c656280e5a46c34e89f301b7a), MIT licence.
`cuda/common/libm_cr.cuh` is derived from `src/binary32/sin/sinf.c`, `cos/cosf.c`, `tan/tanf.c`, `atan2/atan2f.c` and
`pow/powf.c`, Copyright (c) 2022-2025 Alexei Sibidanov (powf with Paul Zimmermann). The licence text is kept in the header of
that file:

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
    Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
    WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
