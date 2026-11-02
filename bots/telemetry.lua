-- Lossless column transport: avoid quoting the same feature names per unit.
local json=require("game/dkjson")
local schema=require("bots/telemetry_schema")
local M={}
local last_skills=nil
local packed_skills=nil
local function row(value,columns)
    local result={}
    for i,key in ipairs(columns) do
        local v=value[key]
        result[i]=v==nil and json.null or v
    end
    return result
end
local function rows(values,columns)
    local result={}
    for i,value in ipairs(values) do result[i]=row(value,columns) end
    return result
end
function M.pack(s)
    local wire={}
    for key,value in pairs(s) do wire[key]=value end
    wire.codec="sf-columns-1"
    wire.self=row(s.self,schema.unit)
    if s.enemy then wire.enemy=row(s.enemy,schema.unit) end
    wire.creeps=rows(s.creeps,schema.unit);wire.towers=rows(s.towers,schema.unit)
    wire.probe={}
    for key,value in pairs(s.probe) do wire.probe[key]=value end
    if last_skills~=s.probe.skill_slots then
        last_skills=s.probe.skill_slots
        packed_skills=rows(s.probe.skill_slots,schema.skills)
        local encoded=json.encode(packed_skills)
        setmetatable(packed_skills,{__tojson=function() return encoded end})
    end
    wire.probe.skill_slots=packed_skills
    wire.probe.inventory=rows(s.probe.inventory,schema.inventory)
    return wire
end
return M
