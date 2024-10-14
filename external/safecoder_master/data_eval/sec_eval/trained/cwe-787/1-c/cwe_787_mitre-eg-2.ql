/**
 * @name Check for CWE-787 in mitre-eg-2
 * @description 
 * @kind problem
 * @problem.severity warning
 * @id custom-cpp/cwe-787-mitre-eg-2
 */

import cpp
import semmle.code.cpp.controlflow.Guards
import semmle.code.cpp.rangeanalysis.SimpleRangeAnalysis
import semmle.code.cpp.controlflow.SSA

predicate lesser_than_size(Expr expr){
    expr.toString().toInt() <= 10
    // if it is a call to returnChunkSize(), the index is automatically lesser than size
    or exists (FunctionCall fc | 
        fc = expr and expr.toString() = "call to returnChunkSize")
    //if it calls chunk[0] or chunk2[0], that also returns the size value
    or exists( ArrayExpr rhs_ae | 
        rhs_ae = expr 
        and ( rhs_ae.getArrayBase().toString() = "chunk"
            or rhs_ae.getArrayBase().toString() = "chunk2" ) 
        and rhs_ae.getArrayOffset().toString() = "0"
        )
    //if it refers to size variable, check the value of size variable
}

predicate size_variable_less_than_size(Expr expr){
    expr.toString() = "size" or expr.toString() = "size2"

    and exists( Expr expr_def, Expr expr_use, Variable size_variable | 
        size_variable.toString() = expr.toString()
        and exists (SsaDefinition ssaDef |
            expr_def = ssaDef.getAnUltimateDefiningValue(size_variable)
            and expr_use = ssaDef.getAUse(size_variable) and 
            expr_use = expr)
        and lesser_than_size(expr_def)
        )

}

from ArrayExpr ae

where

//get the array expressions that are being written into
exists(AssignExpr assign_expr | assign_expr.getLValue() = ae )

//array expression's offset index should not be 0 (those are  used to set sizes)
and not ae.getArrayOffset().toString() = "0" 

//eliminate the array accesses that have correct bounds
and not exists( GuardCondition gc, Expr e, Expr expr |
        //get the array expressions accesses that are guarded with < (some expr)
        gc.ensuresLt(e, expr, 0, ae.getBasicBlock(), true) 

        //offset index should not be using the size variable
        and not (ae.getArrayOffset().toString() = "size" or 
                ae.getArrayOffset().toString() ="size2")

        //ensure that the rhs of the ( < ) guard condition is 10 or lesser
        and ( lesser_than_size(expr) or 
        //if the rhs of gc is the size variable, ensure that its definition
        //gives it a value of less than 10
            size_variable_less_than_size(expr) )
        )

select ae, "cwe_787 found in"+ae.getFile().getShortName()
