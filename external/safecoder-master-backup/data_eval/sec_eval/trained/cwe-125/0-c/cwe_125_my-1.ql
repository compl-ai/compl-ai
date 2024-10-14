/**
 * @name Check for CWE-125 in my1
 * @description 
 * @kind problem
 * @problem.severity warning
 * @id custom-cpp/cwe-125-my1
 */

import cpp
import semmle.code.cpp.controlflow.Guards
import semmle.code.cpp.rangeanalysis.SimpleRangeAnalysis

from ArrayExpr ae, ArrayType ar_t, int array_size

where

ar_t = ae.getArrayBase().getType()
and array_size = ar_t.getArraySize()

and not (lowerBound(ae.getArrayOffset()) >= 0 
    and upperBound( ae.getArrayOffset() ) < array_size
    )

select ae, "cwe_125 found in"+ae.getFile().toString()