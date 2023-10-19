import { useContext, useEffect, useState } from 'react'

import { ToastContext } from 'contexts'
import { useModal } from 'hooks'

import { useFormik } from 'formik'
import { useNavigate } from 'react-router-dom'
import { useSchedulesService } from 'services/schedule/useSchedulesService'
import { useCreateScheduleService } from 'services/schedule/useCreateScheduleService'
import { scheduleValidationSchema } from 'utils/validationsSchema'

export const useCreateSchedule = () => {
  const navigate = useNavigate()

  const { setToast } = useContext(ToastContext)

  const [isLoading, setIsLoading] = useState(false)

  const [createScheduleService] = useCreateScheduleService()

  const { data: schedule, refetch: refetchSchedule } = useSchedulesService()

  const initialValues = {
    schedule_name: '',
    schedule_description: '',
    schedule_is_active: true,
    schedule_max_daily_budget: 0.1,
    schedule_cron_expression: '* * * * *',
    schedule_type: 'Run tasks',
    schedule_agent_id: null,
    schedule_group_id: null,

    agent_type: '',
    tasks: ['Enter you task'],
    run_immediately: false,
    create_session_on_run: false,
  }

  const handleSubmit = async (values: any) => {
    setIsLoading(true)

    const { agent_type } = values

    try {
      await createScheduleService({
        name: values.schedule_name,
        description: values.schedule_description,
        is_active: values.schedule_is_active,
        max_daily_budget: values.schedule_max_daily_budget,
        cron_expression: values.schedule_cron_expression,
        schedule_type: values.schedule_type,
        agent_id: agent_type === 'agent' ? values.schedule_agent_id : null,
        team_id: agent_type === 'team' ? values.schedule_agent_id : null,
        chat_id: agent_type === 'chat' ? values.schedule_agent_id : null,
        group_id: values.schedule_group_id,
        create_session_on_run: values.create_session_on_run,
        run_immediately: values.run_immediately,
        tasks: values.tasks,
      })

      await refetchSchedule()
      setToast({
        message: 'New Schedule was Created!',
        type: 'positive',
        open: true,
      })
      navigate('/schedules')
    } catch (e) {
      setToast({
        message: 'Failed To Add Schedule!',
        type: 'negative',
        open: true,
      })
    }
    setIsLoading(false)
  }

  const formik = useFormik({
    initialValues: initialValues,
    validationSchema: scheduleValidationSchema,
    onSubmit: async values => handleSubmit(values),
  })

  return {
    schedule,
    formik,
    isLoading,
  }
}
