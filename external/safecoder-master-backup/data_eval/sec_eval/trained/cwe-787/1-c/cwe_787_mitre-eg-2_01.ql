import cpp
import semmle.code.cpp.controlflow.Guards
import semmle.code.cpp.rangeanalysis.SimpleRangeAnalysis
import semmle.code.cpp.controlflow.SSA

from ArrayExpr ae

where

//get the array expressions that are being written into
exists(AssignExpr assign_expr | assign_expr.getLValue() = ae )

//array expression's offset index should not be 0 (those are  used to set sizes)
and not ae.getArrayOffset().toString() = "0" 

and exists( GuardCondition gc, Expr e, Expr expr |
    //get the array expressions accesses that are guarded with < (some expr)
    gc.ensuresLt(e, expr, 0, ae.getBasicBlock(), true)

    and (
    
        exists(FunctionCall fc | 

            //the rhs expression of the gc  is the function call returnChunkSize()
            (expr = fc and fc.getTarget().getName() = "returnChunkSize")
             
            or 
            // or the rhs expression of the gc is a variable that is defined by the function call returnChunkSize()
            exists( Expr expr_def, Expr expr_use, Variable size_variable | 
                size_variable.toString() = expr.toString()
                and exists (SsaDefinition ssaDef |
                    expr_def = ssaDef.getAnUltimateDefiningValue(size_variable)
                    and expr_use = ssaDef.getAUse(size_variable) 
                    and expr_use = expr
                    and expr_def = fc
                    and fc.getTarget().getName() = "returnChunkSize"
                    )
            )

        )

    )

)

select ae, "cwe_787 found in"+ae.getFile().getShortName()