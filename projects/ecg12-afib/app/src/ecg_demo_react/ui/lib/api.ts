import { useQuery, useSuspenseQuery, useMutation } from "@tanstack/react-query";
import type { UseQueryOptions, UseSuspenseQueryOptions, UseMutationOptions } from "@tanstack/react-query";
export class ApiError extends Error {
    status: number;
    statusText: string;
    body: unknown;
    constructor(status: number, statusText: string, body: unknown){
        super(`HTTP ${status}: ${statusText}`);
        this.name = "ApiError";
        this.status = status;
        this.statusText = statusText;
        this.body = body;
    }
}
export interface AnalyzeRequest {
    ecg: number[][];
}
export interface AnalyzeResponse {
    afib_label?: string | null;
    afib_model?: string | null;
    arr_flag: number;
    arr_model: string;
    arr_prob: number;
    flagged: boolean;
    p_afib?: number | null;
}
export interface AppConfigOut {
    afib_endpoint: string;
    afib_model: string;
    arr_endpoint: string;
    arr_model: string;
    catalog: string;
    champion_alias: string;
    daily_job_id: string | null;
    dashboard_id: string | null;
    inference_table: string;
    n_leads: number;
    serving_timeout_s: number;
    uc_schema: string;
    warmup_interval_s: number;
    workspace_host: string;
    workspace_id: string;
}
export interface EndpointStatus {
    endpoint: string;
    message?: string | null;
    seconds: number;
    status: "ready" | "cold" | "error";
}
export interface GenerateRequest {
    rhythm?: string;
}
export interface GenerateResponse {
    ecg: number[][];
    label_name: string;
    lead_names: string[];
    n_samples: number;
    patient: PatientRecord;
    sampling_rate: number;
    true_label: number;
}
export interface HTTPValidationError {
    detail?: ValidationError[];
}
export interface PatientRecord {
    acquisition_datetime_utc: string;
    age_years: number;
    bmi: number;
    bp_diastolic_mmhg: number;
    bp_systolic_mmhg: number;
    duration_s: number;
    heart_rate_bpm: number;
    height_cm: number;
    mrn: string;
    n_leads: number;
    pseudo_patient_id: string;
    sampling_rate_hz: number;
    sex: string;
    weight_kg: number;
}
export interface ServingError {
    endpoint: string;
    message: string;
    reason: "waking" | "rate_limited" | "error";
}
export interface ServingErrorOut {
    detail: ServingError;
}
export interface ValidationError {
    ctx?: Record<string, unknown>;
    input?: unknown;
    loc: (string | number)[];
    msg: string;
    type: string;
}
export interface WarmupResponse {
    endpoints: EndpointStatus[];
}
export const analyze = async (data: AnalyzeRequest, options?: RequestInit): Promise<{
    data: AnalyzeResponse;
}> =>{
    const res = await fetch("/api/analyze", {
        ...options,
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            ...options?.headers
        },
        body: JSON.stringify(data)
    });
    if (!res.ok) {
        const body = await res.text();
        let parsed: unknown;
        try {
            parsed = JSON.parse(body);
        } catch  {
            parsed = body;
        }
        throw new ApiError(res.status, res.statusText, parsed);
    }
    return {
        data: await res.json()
    };
};
export function useAnalyze(options?: {
    mutation?: UseMutationOptions<{
        data: AnalyzeResponse;
    }, ApiError, AnalyzeRequest>;
}) {
    return useMutation({
        mutationFn: (data)=>analyze(data),
        ...options?.mutation
    });
}
export const appConfig = async (options?: RequestInit): Promise<{
    data: AppConfigOut;
}> =>{
    const res = await fetch("/api/app-config", {
        ...options,
        method: "GET"
    });
    if (!res.ok) {
        const body = await res.text();
        let parsed: unknown;
        try {
            parsed = JSON.parse(body);
        } catch  {
            parsed = body;
        }
        throw new ApiError(res.status, res.statusText, parsed);
    }
    return {
        data: await res.json()
    };
};
export const appConfigKey = ()=>{
    return [
        "/api/app-config"
    ] as const;
};
export function useAppConfig<TData = {
    data: AppConfigOut;
}>(options?: {
    query?: Omit<UseQueryOptions<{
        data: AppConfigOut;
    }, ApiError, TData>, "queryKey" | "queryFn">;
}) {
    return useQuery({
        queryKey: appConfigKey(),
        queryFn: ()=>appConfig(),
        ...options?.query
    });
}
export function useAppConfigSuspense<TData = {
    data: AppConfigOut;
}>(options?: {
    query?: Omit<UseSuspenseQueryOptions<{
        data: AppConfigOut;
    }, ApiError, TData>, "queryKey" | "queryFn">;
}) {
    return useSuspenseQuery({
        queryKey: appConfigKey(),
        queryFn: ()=>appConfig(),
        ...options?.query
    });
}
export const generate = async (data: GenerateRequest, options?: RequestInit): Promise<{
    data: GenerateResponse;
}> =>{
    const res = await fetch("/api/generate", {
        ...options,
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            ...options?.headers
        },
        body: JSON.stringify(data)
    });
    if (!res.ok) {
        const body = await res.text();
        let parsed: unknown;
        try {
            parsed = JSON.parse(body);
        } catch  {
            parsed = body;
        }
        throw new ApiError(res.status, res.statusText, parsed);
    }
    return {
        data: await res.json()
    };
};
export function useGenerate(options?: {
    mutation?: UseMutationOptions<{
        data: GenerateResponse;
    }, ApiError, GenerateRequest>;
}) {
    return useMutation({
        mutationFn: (data)=>generate(data),
        ...options?.mutation
    });
}
export const warmup = async (options?: RequestInit): Promise<{
    data: WarmupResponse;
}> =>{
    const res = await fetch("/api/warmup", {
        ...options,
        method: "POST"
    });
    if (!res.ok) {
        const body = await res.text();
        let parsed: unknown;
        try {
            parsed = JSON.parse(body);
        } catch  {
            parsed = body;
        }
        throw new ApiError(res.status, res.statusText, parsed);
    }
    return {
        data: await res.json()
    };
};
export function useWarmup(options?: {
    mutation?: UseMutationOptions<{
        data: WarmupResponse;
    }, ApiError, void>;
}) {
    return useMutation({
        mutationFn: ()=>warmup(),
        ...options?.mutation
    });
}
